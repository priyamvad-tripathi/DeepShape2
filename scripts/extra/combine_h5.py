# %%
import time

import h5py

from deepshape2.utils import load_config, time_string

start = time.time()

DIR = load_config()["DATA_DIR"]

file1 = DIR + "training_set_2.h5"
file2 = DIR + "training_set.h5"
outfile = DIR + "training_set_52_100.h5"

with (
    h5py.File(file1, "r") as f1,
    h5py.File(file2, "r") as f2,
    h5py.File(outfile, "w") as fout,
):
    for key in f1.keys():
        print("Combining key:", key)

        d1 = f1[key]
        d2 = f2[key]

        # sanity check on shapes except first axis
        if d1.shape[1:] != d2.shape[1:]:
            raise ValueError(f"Shape mismatch for key {key}: {d1.shape} vs {d2.shape}")

        combined_shape = (d1.shape[0] + d2.shape[0],) + d1.shape[1:]

        d_out = fout.create_dataset(
            key,
            shape=combined_shape,
            maxshape=(None,) + d1.shape[1:],
            dtype=d1.dtype,
            chunks=(64,) + d1.shape[1:],
        )

        d_out[: d1.shape[0]] = d1[:]
        d_out[d1.shape[0] :] = d2[:]

        print("Combibed shape for key", key, ":", d_out.shape)
        print("Time elapsed:", time_string(time.time() - start))
print("All done! Total time:", time_string(time.time() - start))
