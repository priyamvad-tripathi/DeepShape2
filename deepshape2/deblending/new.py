# %% Import Libraries
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from colorist import Color

from deepshape2.utils import (
    get_progress_bar,
    load_ckp,
    psnr_torch,
    save_ckp,
    time_string,
)

# ==========================================================================
# 1. WEIGHTS / MASKS
# ==========================================================================
_MASK_CACHE = {}


def circ_mask(height, width, radius=None, device=None, dtype=torch.float32, soft=False):
    """
    Central weight map.

    radius=None -> min(h, w) / 4  (32 px at 128x128, ~13" at 0.4"/px).

    soft=False : binary disc (hard edge).
    soft=True  : Gaussian of sigma = radius/2, normalised to peak 1.  Smoother
                 and closer to the weighting a moment estimator applies, so the
                 loss and the shape metric agree about which pixels matter.
    """
    if radius is None:
        radius = min(height, width) / 4.0
    y = torch.arange(height, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(width, device=device, dtype=dtype).view(1, -1)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    if soft:
        m = torch.exp(-r2 / (2 * (radius / 2.0) ** 2))
    else:
        m = (torch.sqrt(r2) <= radius).to(dtype)
    return m.to(dtype).view(1, 1, height, width)


def get_circ_mask(height, width, device, dtype, radius=None, soft=False):
    """Cached lookup; broadcasting removes the need for a per-batch copy."""
    key = (height, width, str(device), dtype, radius, soft)
    m = _MASK_CACHE.get(key)
    if m is None:
        m = circ_mask(height, width, radius, device, dtype, soft)
        _MASK_CACHE[key] = m
    return m


# ==========================================================================
# 2. LOSS
# ==========================================================================
def vae_loss(
    target,
    recon,
    mu,
    logvar,
    beta,
    device=None,
    alpha=0.85,
    mask=None,
    mask_radius=None,
    soft_mask=False,
):
    """
    Beta-VAE loss, sum reduction (unchanged from the configuration that reached
    58 dB, so alpha / beta / LR all transfer).

    alpha now defaults to 0.85 (was 0.7): with mask_radius=None the central
    region is 19.6% of the frame rather than 78.5%, so more weight is needed to
    put comparable emphasis on the pixels ShapeNet actually measures.

    Returns (total, recon_sum, kl_sum), same meaning as before.
    """
    if mask is None:
        mask = get_circ_mask(
            target.shape[2],
            target.shape[3],
            target.device,
            target.dtype,
            mask_radius,
            soft_mask,
        )

    recon_loss = F.mse_loss(target, recon, reduction="sum")
    central_loss = F.mse_loss(target * mask, recon * mask, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss * (1 - alpha) + central_loss * alpha + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ==========================================================================
# 3. SHAPE + BLENDEDNESS METRICS
# ==========================================================================
def moments(img, sigma_w=4.0, eps=1e-30):
    """
    Gaussian-weighted quadrupole moments of a LINEAR-FLUX image.

    Returns (e1, e2, T) in the distortion convention, per sample.  Flux-
    invariant by construction: multiplying img by a constant leaves e1, e2 and
    T unchanged, which is what you want when the deblender is allowed to get
    the amplitude wrong but not the shape.

    The weight is centred on the stamp centre, not on a measured centroid, so
    a centroid offset shows up as a shape error -- deliberate, since ShapeNet
    sees the same stamps.
    """
    B, C, H, W = img.shape
    y = (
        torch.arange(H, device=img.device, dtype=img.dtype).view(1, 1, -1, 1)
        - (H - 1) / 2
    )
    x = (
        torch.arange(W, device=img.device, dtype=img.dtype).view(1, 1, 1, -1)
        - (W - 1) / 2
    )
    w = torch.exp(-(x**2 + y**2) / (2 * sigma_w**2))

    I = (img * w).clamp_min(0)
    norm = I.sum((1, 2, 3)).clamp_min(eps)
    Q11 = (I * x**2).sum((1, 2, 3)) / norm
    Q22 = (I * y**2).sum((1, 2, 3)) / norm
    Q12 = (I * x * y).sum((1, 2, 3)) / norm
    T = (Q11 + Q22).clamp_min(eps)
    return (Q11 - Q22) / T, 2 * Q12 / T, T


def blendedness(blend_lin, target_lin, eps=1e-30):
    """
    Per-sample blendedness, b = 1 - (T.T) / (T.B), computed directly from the
    input/target pair -- no catalogue lookup needed.

    b = 0 -> the stamp is isolated (blended image equals the target).
    b -> 1 -> the target's flux is a small fraction of what the stamp contains.
    """
    tt = (target_lin * target_lin).flatten(1).sum(1)
    tb = (target_lin * blend_lin).flatten(1).sum(1)
    return 1.0 - tt / tb.clamp_min(eps)


# ==========================================================================
# 4. EVALUATION
# ==========================================================================
def _decode_from_mu(model, mu):
    """Deterministic decode (z = mu).  None if no usable entry point."""
    for name in ("decode", "decoder"):
        fn = getattr(model, name, None)
        if fn is None:
            continue
        try:
            return fn(mu)
        except Exception:
            continue
    return None


@torch.inference_mode()
def evaluate(
    model,
    val_loader,
    device,
    scale_fac,
    deterministic=True,
    mask_radius=None,
    soft_mask=False,
    sigma_w=4.0,
    iso_thresh=0.01,
    blend_thresh=0.05,
    collect=False,
):
    """
    Full per-sample evaluation.

    Collects, for every validation stamp:
      psnr          global PSNR in linear flux
      psnr_central  PSNR inside the central weight region
      de1, de2      shape error, recon minus target (distortion convention)
      dT            size ratio, T_recon / T_target
      b             blendedness, derived from the input/target pair

    and reports them overall, on the near-isolated subset (b < iso_thresh) and
    on the blended subset (b > blend_thresh).  The isolated subset is the
    acceptance criterion: there the deblender must act as the identity, since
    the correct answer is "change nothing".
    """
    model.eval()
    P, PC, D1, D2, DT, BL = [], [], [], [], [], []
    mu_std_acc, n_batches = 0.0, 0

    for inp, target in val_loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(inp)
        if isinstance(out, (tuple, list)):
            recon, mu = out[0], out[1]
            if deterministic:
                det = _decode_from_mu(model, mu)
                if det is not None:
                    recon = det
            mu_std_acc += float(mu.std(0).mean())
        else:
            recon = out
        n_batches += 1

        # linear flux
        t_lin = torch.sinh(target) / scale_fac
        r_lin = torch.sinh(recon) / scale_fac
        i_lin = torch.sinh(inp) / scale_fac

        P.append(psnr_torch(t_lin, r_lin))

        m = get_circ_mask(
            target.shape[2],
            target.shape[3],
            target.device,
            target.dtype,
            mask_radius,
            soft_mask,
        )
        PC.append(psnr_torch(t_lin * m, r_lin * m))

        e1t, e2t, Tt = moments(t_lin, sigma_w)
        e1r, e2r, Tr = moments(r_lin, sigma_w)
        D1.append(e1r - e1t)
        D2.append(e2r - e2t)
        DT.append(Tr / Tt.clamp_min(1e-30))
        BL.append(blendedness(i_lin, t_lin))

    P, PC = torch.cat(P), torch.cat(PC)
    D1, D2, DT, BL = torch.cat(D1), torch.cat(D2), torch.cat(DT), torch.cat(BL)

    ok = torch.isfinite(P) & torch.isfinite(D1) & torch.isfinite(D2)
    n_dropped = int((~ok).sum())

    def _sub(sel, tag):
        n = int(sel.sum())
        if n == 0:
            return {f"{tag}_n": 0}
        p, pc = P[sel], PC[sel]
        d1, d2, dt = D1[sel], D2[sel], DT[sel]
        dmag = torch.sqrt(d1**2 + d2**2)
        return {
            f"{tag}_n": n,
            f"{tag}_psnr_med": float(p.median()),
            f"{tag}_psnr_p05": float(torch.quantile(p, 0.05)),
            f"{tag}_pc_med": float(pc[torch.isfinite(pc)].median())
            if torch.isfinite(pc).any()
            else float("nan"),
            f"{tag}_e1_bias": float(d1.mean()),
            f"{tag}_e2_bias": float(d2.mean()),
            f"{tag}_e_rms": float(torch.sqrt((d1**2 + d2**2).mean())),
            f"{tag}_e_med": float(dmag.median()),
            f"{tag}_T_ratio_med": float(dt.median()),
        }

    iso = ok & (BL < iso_thresh)
    bln = ok & (BL > blend_thresh)

    stats = {
        "n_samples": int(ok.sum()),
        "n_dropped": n_dropped,
        "mu_std": mu_std_acc / max(n_batches, 1),
        "frac_isolated": float((BL[ok] < iso_thresh).float().mean()),
        "frac_blended": float((BL[ok] > blend_thresh).float().mean()),
        "b_med": float(BL[ok].median()),
        "b_p95": float(torch.quantile(BL[ok], 0.95)),
    }
    stats.update(_sub(ok, "all"))
    stats.update(_sub(iso, "iso"))
    stats.update(_sub(bln, "bln"))

    if collect:
        stats["_per_sample"] = dict(
            psnr=P.cpu(),
            psnr_central=PC.cpu(),
            de1=D1.cpu(),
            de2=D2.cpu(),
            dT=DT.cpu(),
            b=BL.cpu(),
        )
    return stats


@torch.inference_mode()
def validation_loss(
    model,
    val_loader,
    device,
    scale_fac,
    deterministic=True,
    return_stats=False,
    mask_radius=None,
    soft_mask=False,
):
    """
    Mean negative PSNR (lower is better) -- the scheduler / checkpoint signal.

    Kept lightweight for the per-epoch path.  Non-finite values are dropped:
    an unblended stamp can give ~zero MSE -> inf, which poisons the mean.
    """
    model.eval()
    P, PC, D1, D2 = [], [], [], []

    for inp, target in val_loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(inp)
        if isinstance(out, (tuple, list)):
            recon, mu = out[0], out[1]
            if deterministic:
                det = _decode_from_mu(model, mu)
                if det is not None:
                    recon = det
        else:
            recon = out

        t_lin = torch.sinh(target) / scale_fac
        r_lin = torch.sinh(recon) / scale_fac
        P.append(psnr_torch(t_lin, r_lin))

        if return_stats:
            m = get_circ_mask(
                target.shape[2],
                target.shape[3],
                target.device,
                target.dtype,
                mask_radius,
                soft_mask,
            )
            PC.append(psnr_torch(t_lin * m, r_lin * m))
            e1t, e2t, _ = moments(t_lin)
            e1r, e2r, _ = moments(r_lin)
            D1.append(e1r - e1t)
            D2.append(e2r - e2t)

    P = torch.cat(P)
    ok = torch.isfinite(P)
    val = float(-P[ok].mean()) if ok.any() else float("nan")

    if not return_stats:
        return val

    p = P[ok]
    PC = torch.cat(PC)
    pc = PC[torch.isfinite(PC)]
    D1, D2 = torch.cat(D1)[ok], torch.cat(D2)[ok]
    stats = {
        "psnr_mean": float(p.mean()),
        "psnr_med": float(p.median()),
        "psnr_p05": float(torch.quantile(p, 0.05)),
        "pc_med": float(pc.median()) if pc.numel() else float("nan"),
        "e_rms": float(torch.sqrt((D1**2 + D2**2).mean())),
        "e1_bias": float(D1.mean()),
        "e2_bias": float(D2.mean()),
        "n_dropped": int((~ok).sum()),
    }
    return val, stats


class IdentityDeblender(torch.nn.Module):
    """Baseline: returns the blended input unchanged, with the VAE interface."""

    def __init__(self, latent_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        recon = x.clone()
        z = torch.zeros(x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype)
        return recon, z, z


@torch.no_grad()
def recalibrate_bn(model, loader, device, n_batches=200):
    """Reset and re-estimate BatchNorm running statistics before eval."""
    bns = [
        m
        for m in model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    ]
    if not bns:
        return model
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, (inp, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(inp.to(device, non_blocking=True))
    return model


def report_evaluation(
    model,
    val_loader,
    device,
    scale_fac,
    baseline=None,
    mask_radius=None,
    soft_mask=False,
    sigma_w=4.0,
    label="",
    **kw,
):
    """Print a stratified evaluation, optionally against a baseline."""
    st = evaluate(
        model,
        val_loader,
        device,
        scale_fac,
        mask_radius=mask_radius,
        soft_mask=soft_mask,
        sigma_w=sigma_w,
        **kw,
    )

    print(f"\n=== EVAL {label} ===")
    print(
        f"  n={st['n_samples']} dropped={st['n_dropped']} | "
        f"blendedness med {st['b_med']:.4f} p95 {st['b_p95']:.4f} | "
        f"isolated {st['frac_isolated']:.1%} blended {st['frac_blended']:.1%}"
    )
    hdr = (
        f"  {'subset':<10}{'n':>8}{'PSNR med':>10}{'p05':>8}{'central':>10}"
        f"{'|de| med':>11}{'de rms':>10}{'e1 bias':>10}{'e2 bias':>10}{'T ratio':>9}"
    )
    print(hdr)
    for tag, name in (("all", "all"), ("iso", "isolated"), ("bln", "blended")):
        if st.get(f"{tag}_n", 0) == 0:
            continue
        print(
            f"  {name:<10}{st[f'{tag}_n']:>8}{st[f'{tag}_psnr_med']:>10.2f}"
            f"{st[f'{tag}_psnr_p05']:>8.2f}{st[f'{tag}_pc_med']:>10.2f}"
            f"{st[f'{tag}_e_med']:>11.2e}{st[f'{tag}_e_rms']:>10.2e}"
            f"{st[f'{tag}_e1_bias']:>10.2e}{st[f'{tag}_e2_bias']:>10.2e}"
            f"{st[f'{tag}_T_ratio_med']:>9.3f}"
        )

    if baseline is not None:
        print("  vs baseline:")
        for tag, name in (("all", "all"), ("iso", "isolated"), ("bln", "blended")):
            if st.get(f"{tag}_n", 0) == 0 or baseline.get(f"{tag}_n", 0) == 0:
                continue
            dp = st[f"{tag}_psnr_med"] - baseline[f"{tag}_psnr_med"]
            dc = st[f"{tag}_pc_med"] - baseline[f"{tag}_pc_med"]
            re = st[f"{tag}_e_rms"] / max(baseline[f"{tag}_e_rms"], 1e-30)
            print(
                f"    {name:<10} PSNR {dp:+7.2f} dB | central {dc:+7.2f} dB | "
                f"de rms x{re:.2f}"
            )
    return st


# ==========================================================================
# 5. TRAINING
# ==========================================================================
def train(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    device,
    **kwargs,
):
    start_time = time.time()
    best_val_loss = np.inf
    best_weights = None
    current_epoch = 0
    best_epoch = 0
    is_best = False

    train_loss_list, recon_loss_list, kl_loss_list, val_loss_list, lr_list = (
        [],
        [],
        [],
        [],
        [np.inf],
    )
    psnr_med_list, pc_med_list, e_rms_list = [], [], []

    # --- Config ---
    filename = kwargs.get("filename")
    scheduler_params = kwargs.get("scheduler_params")
    save_freq = kwargs.get("save_freq", 50)
    beta = kwargs.get("beta", 1e-4)
    precision = kwargs.get("precision", 4)
    tqdm_kwargs = kwargs.get("tqdm_kwargs", dict(colour="green", unit="batch"))
    tqdm_flag = kwargs.get("tqdm_flag", True)
    alpha = kwargs.get("alpha", 0.85)
    mask_radius = kwargs.get("mask_radius", None)  # None -> min(h,w)/4
    soft_mask = kwargs.get("soft_mask", False)
    clip_grad = kwargs.get("clip_grad", None)
    deterministic_val = kwargs.get("deterministic_val", True)
    scale_fac = kwargs.get("scale_fac", 5e7)

    scheduler = None
    if scheduler_params:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )

    print(f"Running on device: {device}")
    print(
        f"beta={beta} | alpha={alpha} | mask_radius={mask_radius} | "
        f"soft_mask={soft_mask} | clip_grad={clip_grad}"
    )

    # --- Load checkpoint ---
    try:
        model, optimizer, checkpoint = load_ckp(filename, model, optimizer, device)
        current_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", np.inf)
        best_weights = checkpoint.get("best_weights")
        best_epoch = checkpoint.get("best_epoch", 0)
        val_loss_list = checkpoint.get("val_loss_list", [])
        recon_loss_list = checkpoint.get("recon_loss_list", [])
        kl_loss_list = checkpoint.get("kl_loss_list", [])
        train_loss_list = checkpoint.get("train_loss_list", [])
        psnr_med_list = checkpoint.get("psnr_med_list", [])
        pc_med_list = checkpoint.get("pc_med_list", [])
        e_rms_list = checkpoint.get("e_rms_list", [])
        lr_list = checkpoint.get("lr_list", [])
        if not lr_list:
            lr_list = [np.inf]
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}")
    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found. Starting from scratch.")

    print(f"scale_fac={scale_fac:.2e}")

    # --- Training loop ---
    for epoch in range(epochs):
        if epoch < current_epoch:
            continue

        model.train()
        sum_loss = sum_recon = sum_kl = sum_gnorm = 0.0
        min_gnorm = np.inf
        n_batches = 0

        current_lr = optimizer.param_groups[0]["lr"]
        new_lr = bool(lr_list) and current_lr < lr_list[-1]
        lr_list.append(current_lr)

        if not tqdm_flag:
            print(
                f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.2e}"
                + (" NEW" if new_lr else ""),
                flush=True,
            )

        pbar = get_progress_bar(tqdm_flag, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        with pbar:
            for inp, target in train_loader:
                inp, target = (
                    inp.to(device, non_blocking=True),
                    target.to(device, non_blocking=True),
                )

                optimizer.zero_grad(set_to_none=True)
                out = model(inp)
                if isinstance(out, (tuple, list)):
                    loss, recon, kl = vae_loss(
                        target,
                        *out[:3],
                        beta=beta,
                        alpha=alpha,
                        mask_radius=mask_radius,
                        soft_mask=soft_mask,
                    )
                else:
                    recon = F.mse_loss(out, target)
                    kl = torch.tensor(0.0, device=device)
                    loss = recon

                loss.backward()

                # Gradient norm is measured every step but only summarised per
                # epoch.  A zero here is the one signal that catches a dead
                # output head, which no loss curve would reveal.
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_grad if clip_grad else float("inf")
                )
                optimizer.step()

                g = float(gnorm)
                sum_loss += loss.detach().item()
                sum_recon += recon.item()
                sum_kl += kl.item()
                sum_gnorm += g
                min_gnorm = min(min_gnorm, g)
                n_batches += 1

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Loss": f"{sum_loss / n_batches:.{precision}e}",
                        "Recon": f"{sum_recon / n_batches:.{precision}e}",
                        "KL": f"{sum_kl / n_batches:.{precision}e}",
                        "LR": (
                            f"{Color.RED}{current_lr:.2e}{Color.OFF}"
                            if new_lr
                            else f"{current_lr:.2e}"
                        ),
                    }
                )

            nb = max(n_batches, 1)
            epoch_loss = sum_loss / nb
            epoch_recon = sum_recon / nb
            epoch_kl = sum_kl / nb
            epoch_gnorm = sum_gnorm / nb

            train_loss_list.append(epoch_loss)
            recon_loss_list.append(epoch_recon)
            kl_loss_list.append(epoch_kl)

            print(
                f"Train Loss: {epoch_loss:.{precision}e} | "
                f"Recon: {epoch_recon:.{precision}e} | "
                f"KL: {epoch_kl:.{precision}e} | |g| {epoch_gnorm:.2e}",
                flush=True,
            )
            if min_gnorm == 0.0:
                print(
                    f"{Color.RED}  WARNING: gradient norm hit exactly zero -- "
                    f"check the output activation for saturation.{Color.OFF}",
                    flush=True,
                )

            # --- Validation ---
            if val_loader:
                val_loss, vs = validation_loss(
                    model,
                    val_loader,
                    device=device,
                    scale_fac=scale_fac,
                    deterministic=deterministic_val,
                    return_stats=True,
                    mask_radius=mask_radius,
                    soft_mask=soft_mask,
                )
                val_loss_list.append(val_loss)
                psnr_med_list.append(vs["psnr_med"])
                pc_med_list.append(vs["pc_med"])
                e_rms_list.append(vs["e_rms"])

                if scheduler:
                    scheduler.step(val_loss)

                is_best = val_loss < best_val_loss
                if is_best:
                    best_epoch = epoch
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

                marker = f" {Color.RED}BEST{Color.OFF}" if is_best else ""
                print(
                    f"  val: mean {vs['psnr_mean']:.2f} | med {vs['psnr_med']:.2f} | "
                    f"p05 {vs['psnr_p05']:.2f} | central {vs['pc_med']:.2f} dB{marker}",
                    flush=True,
                )
                print(
                    f"  shape: |de| rms {vs['e_rms']:.3e} | "
                    f"e1 bias {vs['e1_bias']:+.3e} | e2 bias {vs['e2_bias']:+.3e} | "
                    f"dropped {vs['n_dropped']}",
                    flush=True,
                )

                pbar.set_postfix(
                    {
                        "Loss": f"{epoch_loss:.{precision}e}",
                        "Val": f"{-val_loss:.2f} dB",
                        "|de|": f"{vs['e_rms']:.2e}",
                    }
                )
            else:
                is_best = False
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

            print(f"  elapsed {time_string(time.time() - start_time)}", flush=True)
            print("-" * 60, flush=True)

        # Save checkpoint
        if filename:
            is_final_epoch = (epoch + 1) == epochs
            is_save_epoch = (epoch + 1) % save_freq == 0
            if is_final_epoch or is_save_epoch or is_best:
                checkpoint_data = {
                    "epoch": epoch + 1,
                    "model": model,
                    "optimizer": optimizer,
                    "best_weights": best_weights,
                    "filename": filename,
                    "best_val_loss": best_val_loss,
                    "val_loss_list": val_loss_list,
                    "recon_loss_list": recon_loss_list,
                    "kl_loss_list": kl_loss_list,
                    "beta": beta,
                    "train_loss_list": train_loss_list,
                    "lr_list": lr_list[1:],
                    "best_epoch": best_epoch,
                    "alpha": alpha,
                    "mask_radius": mask_radius,
                    "soft_mask": soft_mask,
                    "scale_fac": scale_fac,
                    "psnr_med_list": psnr_med_list,
                    "pc_med_list": pc_med_list,
                    "e_rms_list": e_rms_list,
                }
                if scheduler:
                    checkpoint_data["scheduler_state_dict"] = copy.deepcopy(
                        scheduler.state_dict()
                    )
                print(
                    f"Saving {'final' if is_final_epoch else 'intermediate'} "
                    f"checkpoint at Epoch {epoch + 1} at "
                    f"{time_string(time.time() - start_time)}",
                    flush=True,
                )
                save_ckp(**checkpoint_data)

    total_time = time.time() - start_time
    print("-" * 60)

    if val_loader and val_loss_list:
        bi = min(best_epoch, len(train_loss_list) - 1)
        print(
            f"Training completed in {time_string(total_time)}\n"
            f"Best Val Loss at Epoch {best_epoch + 1}: "
            f"{-min(val_loss_list):.3f} dB mean\n"
            f"  PSNR med {psnr_med_list[bi]:.3f} | central {pc_med_list[bi]:.3f} | "
            f"|de| rms {e_rms_list[bi]:.3e}\n"
            f"  Train={train_loss_list[bi]:.{precision}e}, "
            f"Recon={recon_loss_list[bi]:.{precision}e}, "
            f"KL={kl_loss_list[bi]:.{precision}e}"
        )
    elif train_loss_list:
        best_idx = train_loss_list.index(min(train_loss_list))
        print(
            f"Training completed in {time_string(total_time)}\n"
            f"Best Train Loss at Epoch {best_idx + 1}: "
            f"{min(train_loss_list):.{precision}e}"
        )
    print("-" * 60)
    print(f"Save path: {filename}")

    return best_weights, train_loss_list, val_loss_list
