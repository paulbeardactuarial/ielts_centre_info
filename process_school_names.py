# %%
import langchain_core.prompts
import re
import pandas as pd
from datetime import datetime
import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
import json

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
school_names = school_name_list[:50]

# %%
chunk_size = 100
no_schools = len(school_name_list)
no_chunks = no_schools//chunk_size + 1
# NEED TO WRITE SECTION FOR CHUNKING UP THE LIST


# %%
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# %%
system_prompt = """
you are a helpful chatbot that is great at finding contact details. You will be given a list of UK-based IELTS schools. Find the contact details of each and return in JSON format. Return only for each item the "phone", "email" and "address" fields populated for that school. Return the JSON only.
"""

# %%
# %%
prompt_template = langchain_core.prompts.ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("user", "{school_name_list}")
    ]
)

prompt = prompt_template.invoke({"school_name_list": school_names})


# %%
response = model.invoke(prompt)

# %%

output = response.content
output = output.removeprefix("```json")
output = output.removesuffix("```")
school_data = pd.DataFrame(json.loads(
    output), index=school_names).reset_index(names="school")

# %%
school_data
