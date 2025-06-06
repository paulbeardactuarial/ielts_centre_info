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
prompt = "find me the contact details for the following places. For each place, return a JSON file with the 'phone', 'email' and 'address' fields populated. If you cannot find anything leave as 'None'."
