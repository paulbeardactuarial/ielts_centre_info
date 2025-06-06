# %%
import re
import pandas as pd
from datetime import datetime

# %%
data_raw = pd.read_csv(
    "raw_school_names.txt",
    delimiter="/n",
    header=None,
    names=["school_raw"]
)

# %%

data_raw["school_clean"] = data_raw["school_raw"].str.extract(r"^(.*?),")

# %%
