# %% Imports
import os

import pandas as pd

from deepshape2.utils import load_config, load_h5

# %% Constants / Configuration
cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
RLF_DIR = cfg["RLF_DIR"]

# %%
data = load_h5(os.path.join(DATA_DIR, "deep_set.h5"))
patch_df = pd.DataFrame.from_records(data["patch_000"]["patch_df"][()])
