# %%
import langchain_core.prompts
import re
import pandas as pd
from datetime import datetime
import os
from langchain.chat_models import init_chat_model

# note instructions on setting up langchain can be found here...
# https://python.langchain.com/docs/tutorials/llm_chain/

# %%
data_raw = pd.read_csv(
    "raw_school_names.txt",
    delimiter="/n",
    header=None,
    names=["school_raw"]
)

# %%
data_raw["school_clean"] = data_raw["school_raw"].str.extract(r"^(.*?),")
school_name_list = list(data_raw["school_clean"].dropna().unique())

# %%
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# %%
system_prompt = "you are a helpful and laconic chatbot. You will be given a list of UK-based IELTS schools. Find the contact details of each place. Return only a Python list, where each item is a dictionary file with the 'phone', 'email' and 'address' fields populated for that school"

# %%
# %%
prompt_template = langchain_core.prompts.ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("user", "{school_name_list}")
    ]
)

prompt = prompt_template.invoke(
    {"school_name_list": school_name_list[:10]}
)

# %%
response = model.invoke(prompt)

# %%
