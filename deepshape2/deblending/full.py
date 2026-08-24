import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from deepshape2.utils import (
    get_progress_bar,
    load_ckp,
    psnr_torch,
    save_ckp,
    time_string,
)

# ==========================================================================
# 1. CENTRAL WEIGHT MASK
# ==========================================================================
_MASK_CACHE = {}


def circ_mask(height, width, radius=None, device=None, dtype=torch.float32):
    """Binary central disc. radius=None -> min(h, w) / 2 (64 px at 128x128)."""
    if radius is None:
        radius = min(height, width) / 2.0
    y = torch.arange(height, device=device, dtype=dtype).view(-1, 1)
    x = torch.arange(width, device=device, dtype=dtype).view(1, -1)
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return (torch.sqrt(r2) <= radius).to(dtype).view(1, 1, height, width)


def get_circ_mask(height, width, device, dtype, radius=None):
    key = (height, width, str(device), dtype, radius)
    m = _MASK_CACHE.get(key)
    if m is None:
        m = circ_mask(height, width, radius, device, dtype)
        _MASK_CACHE[key] = m
    return m


# ==========================================================================
# 2. LOSS
# ==========================================================================
def vae_loss(
    target, recon, mu, logvar, beta, device=None, alpha=0.7, mask=None, mask_radius=None
):
    """Beta-VAE loss with an extra central-region term. Sum reduction."""
    if mask is None:
        mask = get_circ_mask(
            target.shape[2], target.shape[3], target.device, target.dtype, mask_radius
        )

    recon_loss = F.mse_loss(target, recon, reduction="sum")
    central_loss = F.mse_loss(target * mask, recon * mask, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss * (1 - alpha) + central_loss * alpha + beta * kl_loss
    return total, recon_loss, kl_loss


# ==========================================================================
# 3. SHAPE + BLENDEDNESS METRICS
# ==========================================================================
def moments(img, sigma_w=4.0, eps=1e-30):
    """
    Gaussian-weighted quadrupole moments of a linear-flux image.
    Returns (e1, e2, T) in the distortion convention, per sample.

    Flux-invariant, and weighted on the stamp centre rather than a measured
    centroid, so a centroid offset registers as a shape error.
    """
    _, _, H, W = img.shape
    y = (
        torch.arange(H, device=img.device, dtype=img.dtype).view(1, 1, -1, 1)
        - (H - 1) / 2
    )
    x = (
        torch.arange(W, device=img.device, dtype=img.dtype).view(1, 1, 1, -1)
        - (W - 1) / 2
    )
    w = torch.exp(-(x**2 + y**2) / (2 * sigma_w**2))

    I_wt = (img * w).clamp_min(0)
    norm = I_wt.sum((1, 2, 3)).clamp_min(eps)
    Q11 = (I_wt * x**2).sum((1, 2, 3)) / norm
    Q22 = (I_wt * y**2).sum((1, 2, 3)) / norm
    Q12 = (I_wt * x * y).sum((1, 2, 3)) / norm
    T = (Q11 + Q22).clamp_min(eps)
    return (Q11 - Q22) / T, 2 * Q12 / T, T


def blendedness(blend_lin, target_lin, eps=1e-30):
    """b = 1 - (T.T)/(T.B), from the input/target pair. b=0 -> isolated."""
    tt = (target_lin * target_lin).flatten(1).sum(1)
    tb = (target_lin * blend_lin).flatten(1).sum(1)
    return 1.0 - tt / tb.clamp_min(eps)


# ==========================================================================
# 4. EVALUATION
# ==========================================================================
def _decode_from_mu(model, mu, inp=None):
    """
    Deterministic decode (z = mu). None if unavailable.

    The mask_mode guard matters: for the mask models model.decoder(mu) returns
    the mask, not an image, and would otherwise be used silently.
    """
    fn = getattr(model, "decode", None)
    if fn is not None:
        if inp is not None:
            try:
                return fn(mu, inp)
            except TypeError:
                pass
            except Exception:
                return None
        try:
            return fn(mu)
        except Exception:
            pass

    if getattr(model, "mask_mode", False):
        return None

    dec = getattr(model, "decoder", None)
    if dec is not None:
        try:
            return dec(mu)
        except Exception:
            pass
    return None


class IdentityDeblender(torch.nn.Module):
    """Baseline: returns the blended input unchanged, with the VAE interface."""

    def __init__(self, latent_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        z = torch.zeros(x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype)
        return x.clone(), z, z


@torch.inference_mode()
def evaluate(
    model,
    val_loader,
    device,
    scale_fac,
    deterministic=True,
    mask_radius=None,
    sigma_w=4.0,
    iso_thresh=0.01,
    blend_thresh=0.05,
    collect=False,
):
    """
    Per-sample evaluation, reported overall and split by blendedness.

    The isolated subset is the acceptance criterion (the deblender should act
    as the identity there); the blended subset is what it exists to fix.
    """
    model.eval()
    P, PC, D1, D2, DT, BL = [], [], [], [], [], []

    for inp, target in val_loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(inp)
        if isinstance(out, (tuple, list)):
            recon, mu = out[0], out[1]
            if deterministic:
                det = _decode_from_mu(model, mu, inp)
                if det is not None:
                    recon = det
        else:
            recon = out

        t_lin = torch.sinh(target) / scale_fac
        r_lin = torch.sinh(recon) / scale_fac
        i_lin = torch.sinh(inp) / scale_fac

        P.append(psnr_torch(t_lin, r_lin))
        m = get_circ_mask(
            target.shape[2], target.shape[3], target.device, target.dtype, mask_radius
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

    def _sub(sel, tag):
        if not sel.any():
            return {f"{tag}_n": 0}
        p, pc = P[sel], PC[sel]
        d1, d2, dt = D1[sel], D2[sel], DT[sel]
        return {
            f"{tag}_n": int(sel.sum()),
            f"{tag}_psnr_med": float(p.median()),
            f"{tag}_psnr_p05": float(torch.quantile(p, 0.05)),
            f"{tag}_pc_med": float(pc[torch.isfinite(pc)].median())
            if torch.isfinite(pc).any()
            else float("nan"),
            f"{tag}_e1_bias": float(d1.mean()),
            f"{tag}_e2_bias": float(d2.mean()),
            f"{tag}_e_rms": float(torch.sqrt((d1**2 + d2**2).mean())),
            f"{tag}_e_med": float(torch.sqrt(d1**2 + d2**2).median()),
            f"{tag}_T_ratio_med": float(dt.median()),
        }

    stats = {
        "n_samples": int(ok.sum()),
        "n_dropped": int((~ok).sum()),
        "frac_isolated": float((BL[ok] < iso_thresh).float().mean()),
        "frac_blended": float((BL[ok] > blend_thresh).float().mean()),
        "b_med": float(BL[ok].median()),
        "b_p95": float(torch.quantile(BL[ok], 0.95)),
    }
    stats.update(_sub(ok, "all"))
    stats.update(_sub(ok & (BL < iso_thresh), "iso"))
    stats.update(_sub(ok & (BL > blend_thresh), "bln"))

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


def report_evaluation(
    model,
    val_loader,
    device,
    scale_fac,
    baseline=None,
    mask_radius=None,
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
        sigma_w=sigma_w,
        **kw,
    )

    print(f"\n=== EVAL {label} ===")
    print(
        f"  n={st['n_samples']} dropped={st['n_dropped']} | "
        f"blendedness med {st['b_med']:.4f} p95 {st['b_p95']:.4f} | "
        f"isolated {st['frac_isolated']:.1%} blended {st['frac_blended']:.1%}"
    )
    print(
        f"  {'subset':<10}{'n':>8}{'PSNR med':>10}{'p05':>8}{'central':>10}"
        f"{'|de| med':>11}{'de rms':>10}{'e1 bias':>10}{'e2 bias':>10}"
        f"{'T ratio':>9}"
    )
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


@torch.inference_mode()
def validation_loss(
    model,
    val_loader,
    device,
    scale_fac,
    deterministic=True,
    return_stats=False,
    mask_radius=None,
    iso_thresh=0.01,
    blend_thresh=0.05,
):
    """
    Mean negative PSNR (lower is better) -- the scheduler signal.

    With return_stats=True, also returns shape metrics split by blendedness.
    Non-finite PSNRs are dropped: a barely-blended stamp can give ~zero MSE.
    """
    model.eval()
    P, PC, D1, D2, BL = [], [], [], [], []

    for inp, target in val_loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        out = model(inp)
        if isinstance(out, (tuple, list)):
            recon, mu = out[0], out[1]
            if deterministic:
                det = _decode_from_mu(model, mu, inp)
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
            )
            PC.append(psnr_torch(t_lin * m, r_lin * m))
            e1t, e2t, _ = moments(t_lin)
            e1r, e2r, _ = moments(r_lin)
            D1.append(e1r - e1t)
            D2.append(e2r - e2t)
            BL.append(blendedness(torch.sinh(inp) / scale_fac, t_lin))

    P = torch.cat(P)
    ok = torch.isfinite(P)
    val = float(-P[ok].mean()) if ok.any() else float("nan")

    if not return_stats:
        return val

    p = P[ok]
    pc = torch.cat(PC)
    pc = pc[torch.isfinite(pc)]

    D1, D2, BL = torch.cat(D1), torch.cat(D2), torch.cat(BL)
    ok2 = ok & torch.isfinite(D1) & torch.isfinite(D2) & torch.isfinite(BL)
    d1, d2, bl = D1[ok2], D2[ok2], BL[ok2]
    de2 = d1**2 + d2**2

    iso, bln = bl < iso_thresh, bl > blend_thresh

    def _rms(sel):
        return float(torch.sqrt(de2[sel].mean())) if sel.any() else float("nan")

    return val, {
        "psnr_mean": float(p.mean()),
        "psnr_med": float(p.median()),
        "psnr_p05": float(torch.quantile(p, 0.05)),
        "pc_med": float(pc.median()) if pc.numel() else float("nan"),
        "e_rms": _rms(torch.ones_like(iso)),
        "e_rms_iso": _rms(iso),
        "e_rms_bln": _rms(bln),
        "e1_bias": float(d1.mean()),
        "e2_bias": float(d2.mean()),
        "e1_bias_bln": float(d1[bln].mean()) if bln.any() else float("nan"),
        "e2_bias_bln": float(d2[bln].mean()) if bln.any() else float("nan"),
        "n_iso": int(iso.sum()),
        "n_bln": int(bln.sum()),
        "n_dropped": int((~ok).sum()),
    }


# ==========================================================================
# 5. CHECKPOINT SELECTION
# ==========================================================================
# name -> (stats key, sign making sign*value LOWER-better)
SELECT_METRICS = {
    "psnr_mean": ("psnr_mean", -1.0),
    "psnr_med": ("psnr_med", -1.0),
    "e_rms": ("e_rms", 1.0),
    "e_rms_iso": ("e_rms_iso", 1.0),
    "e_rms_bln": ("e_rms_bln", 1.0),
}


def _selection_score(stats, select_by):
    key, sign = SELECT_METRICS[select_by]
    v = stats.get(key, float("nan"))
    return float("inf") if not np.isfinite(v) else sign * v


def _improved(score, best, min_rel=0.0):
    """
    True when `score` (lower is better) beats `best` by the required margin.
    The first epoch (best = inf) is handled explicitly: `min_rel * abs(inf)`
    is nan or inf, and comparisons against either are False.
    """
    if not np.isfinite(score):
        return False
    if not np.isfinite(best):
        return True
    if min_rel <= 0:
        return score < best
    return score < best - min_rel * abs(best)


# ==========================================================================
# 6. TRAINING
# ==========================================================================
def train(model, train_loader, val_loader, epochs, optimizer, device, **kwargs):
    """
    kwargs:
      filename, scheduler_params, save_freq, beta, alpha, mask_radius,
      clip_grad, scale_fac, deterministic_val, precision, tqdm_flag,
      select_by      metric driving best_weights (default "psnr_mean")
      min_rel_improve  required fractional gain; scale differs per metric
                       (psnr_mean score ~ -58, e_rms_bln ~ 0.03)
      verbose        adds shape/bias/mask/grad lines to the epoch log
    """
    start_time = time.time()
    best_val_loss = np.inf
    best_score = np.inf
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
    psnr_med_list, pc_med_list = [], []
    e_rms_list, e_rms_iso_list, e_rms_bln_list = [], [], []

    filename = kwargs.get("filename")
    scheduler_params = kwargs.get("scheduler_params")
    save_freq = kwargs.get("save_freq", 50)
    beta = kwargs.get("beta", 1e-4)
    alpha = kwargs.get("alpha", 0.7)
    mask_radius = kwargs.get("mask_radius", None)
    clip_grad = kwargs.get("clip_grad", 1.0)
    scale_fac = kwargs.get("scale_fac", 5e7)
    deterministic_val = kwargs.get("deterministic_val", True)
    precision = kwargs.get("precision", 4)
    tqdm_kwargs = kwargs.get("tqdm_kwargs", dict(colour="green", unit="batch"))
    tqdm_flag = kwargs.get("tqdm_flag", True)
    select_by = kwargs.get("select_by", "psnr_mean")
    min_rel_improve = kwargs.get("min_rel_improve", 0.0)
    verbose = kwargs.get("verbose", False)

    if select_by not in SELECT_METRICS:
        raise ValueError(f"select_by={select_by!r} not in {sorted(SELECT_METRICS)}")

    scheduler = None
    if scheduler_params:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )

    print(
        f"device={device} | beta={beta} | alpha={alpha} | "
        f"mask_radius={mask_radius} | clip_grad={clip_grad} | "
        f"scale_fac={scale_fac:.2e}"
    )
    print(f"select_by={select_by} | min_rel_improve={min_rel_improve}", flush=True)

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
        e_rms_iso_list = checkpoint.get("e_rms_iso_list", [])
        e_rms_bln_list = checkpoint.get("e_rms_bln_list", [])
        lr_list = checkpoint.get("lr_list", []) or [np.inf]

        best_score = checkpoint.get("best_score", np.inf)
        if not np.isfinite(best_score):
            best_score = best_val_loss if select_by == "psnr_mean" else np.inf
        prev_select = checkpoint.get("select_by")
        if (prev_select is not None and prev_select != select_by) or (
            best_weights is None and np.isfinite(best_score)
        ):
            best_score = np.inf  # selection metric changed, or no weights saved

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {current_epoch}", flush=True)
    except (AttributeError, FileNotFoundError, TypeError):
        print("No saved checkpoints found. Starting from scratch.", flush=True)

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

        pbar = get_progress_bar(tqdm_flag, total=len(train_loader), **tqdm_kwargs)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        with pbar:
            for inp, target in train_loader:
                inp = inp.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(inp)
                if isinstance(out, (tuple, list)):
                    loss, recon, kl = vae_loss(
                        target,
                        *out[:3],
                        beta=beta,
                        alpha=alpha,
                        mask_radius=mask_radius,
                    )
                else:
                    recon = F.mse_loss(out, target)
                    kl = torch.tensor(0.0, device=device)
                    loss = recon

                if verbose and n_batches == 0 and hasattr(model, "predict_mask"):
                    with torch.no_grad():
                        mm = model.predict_mask(inp)
                        on = target > 0
                        print(
                            f"  mask: on-source {mm[on].mean():.4f} | "
                            f"off-source {mm[~on].mean():.4f} | "
                            f"frac>0.99 {(mm > 0.99).float().mean():.4f} | "
                            f"min {mm.min():.4f}",
                            flush=True,
                        )

                loss.backward()

                # Measured every step; a zero norm is the only signal that
                # catches a saturated output head.
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
                        "LR": f"{current_lr:.2e}" + ("*" if new_lr else ""),
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

            if min_gnorm == 0.0:
                print(
                    "  !!!!! WARNING: gradient norm hit exactly zero -- "
                    "check the output activation for saturation.",
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
                )
                val_loss_list.append(val_loss)
                psnr_med_list.append(vs["psnr_med"])
                pc_med_list.append(vs["pc_med"])
                e_rms_list.append(vs["e_rms"])
                e_rms_iso_list.append(vs["e_rms_iso"])
                e_rms_bln_list.append(vs["e_rms_bln"])

                if scheduler:
                    # Steps on mean PSNR: smoother than a subset shape metric,
                    # which matters for a plateau detector.
                    scheduler.step(val_loss)

                best_val_loss = min(best_val_loss, val_loss)

                score = _selection_score(vs, select_by)
                is_best = _improved(score, best_score, min_rel_improve)
                if is_best:
                    best_epoch = epoch
                    best_score = score
                    best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

                sel_key = SELECT_METRICS[select_by][0]
                marker = (
                    f"   <<<<< BEST [{select_by}={vs[sel_key]:.4e}]" if is_best else ""
                )

                print(
                    f"Epoch {epoch + 1}/{epochs} | LR {current_lr:.2e}"
                    + ("*" if new_lr else "")
                    + f" | train {epoch_loss:.{precision}e}"
                    + f" | val {vs['psnr_mean']:.2f} dB"
                    + f" | med {vs['psnr_med']:.2f} dB"
                    + f" | {time_string(time.time() - start_time)}"
                    + marker,
                    flush=True,
                )

                if verbose:
                    print(
                        f"  recon {epoch_recon:.{precision}e} | "
                        f"kl {epoch_kl:.{precision}e} | |g| {epoch_gnorm:.2e} | "
                        f"central {vs['pc_med']:.2f} dB | "
                        f"p05 {vs['psnr_p05']:.2f} dB",
                        flush=True,
                    )
                    print(
                        f"  shape: |de| rms {vs['e_rms']:.3e} | "
                        f"iso {vs['e_rms_iso']:.3e} ({vs['n_iso']}) | "
                        f"bln {vs['e_rms_bln']:.3e} ({vs['n_bln']}) | "
                        f"dropped {vs['n_dropped']}",
                        flush=True,
                    )
                    print(
                        f"  bias: e1 {vs['e1_bias']:+.3e} | "
                        f"e2 {vs['e2_bias']:+.3e} | "
                        f"bln e1 {vs['e1_bias_bln']:+.3e} | "
                        f"bln e2 {vs['e2_bias_bln']:+.3e}",
                        flush=True,
                    )

                pbar.set_postfix(
                    {
                        "Loss": f"{epoch_loss:.{precision}e}",
                        "Val": f"{-val_loss:.2f} dB",
                    }
                )
            else:
                is_best = False
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                print(
                    f"Epoch {epoch + 1}/{epochs} | LR {current_lr:.2e} | "
                    f"train {epoch_loss:.{precision}e} | "
                    f"{time_string(time.time() - start_time)}",
                    flush=True,
                )

        # --- Save checkpoint ---
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
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "select_by": select_by,
                    "beta": beta,
                    "alpha": alpha,
                    "mask_radius": mask_radius,
                    "scale_fac": scale_fac,
                    "train_loss_list": train_loss_list,
                    "recon_loss_list": recon_loss_list,
                    "kl_loss_list": kl_loss_list,
                    "val_loss_list": val_loss_list,
                    "lr_list": lr_list[1:],
                    "psnr_med_list": psnr_med_list,
                    "pc_med_list": pc_med_list,
                    "e_rms_list": e_rms_list,
                    "e_rms_iso_list": e_rms_iso_list,
                    "e_rms_bln_list": e_rms_bln_list,
                }
                if scheduler:
                    checkpoint_data["scheduler_state_dict"] = copy.deepcopy(
                        scheduler.state_dict()
                    )
                if verbose:
                    print(f"  saving checkpoint at epoch {epoch + 1}", flush=True)
                save_ckp(**checkpoint_data)

    print("-" * 60)
    if val_loader and val_loss_list:
        bi = min(best_epoch, len(train_loss_list) - 1)
        bv = min(best_epoch, len(val_loss_list) - 1)
        print(
            f"Completed in {time_string(time.time() - start_time)}\n"
            f"Best ({select_by}) at epoch {best_epoch + 1}\n"
            f"  PSNR mean {-val_loss_list[bv]:.3f} | med {psnr_med_list[bi]:.3f} | "
            f"central {pc_med_list[bi]:.3f}\n"
            f"  |de| rms {e_rms_list[bi]:.3e} | iso {e_rms_iso_list[bi]:.3e} | "
            f"bln {e_rms_bln_list[bi]:.3e}"
        )
        if best_weights is None:
            print("  !!!!! WARNING: best_weights is None -- no epoch selected.")
    elif train_loss_list:
        bi = train_loss_list.index(min(train_loss_list))
        print(
            f"Completed in {time_string(time.time() - start_time)}\n"
            f"Best train loss at epoch {bi + 1}: {min(train_loss_list):.{precision}e}"
        )
    print(f"Save path: {filename}")

    return best_weights, train_loss_list, val_loss_list
