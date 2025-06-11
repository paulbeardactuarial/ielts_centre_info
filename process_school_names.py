# %%
import langchain_core.prompts
import re
import pandas as pd
from datetime import datetime
import os
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core import prompts
import json

# note instructions on setting up langchain can be found here...
# https://python.langchain.com/docs/tutorials/llm_chain/


# %%
def llm_response_to_json(content):
    output = content
    output = output.removeprefix("```json")
    output = output.removesuffix("```")
    return (output)


def chunk_my_list(list_to_be_chunked, chunk_size):
    no_items = len(list_to_be_chunked)
    no_chunks = no_items//chunk_size + 1
    master_list = []
    for i in range(no_chunks):
        mini_list = list_to_be_chunked[(
            chunk_size * i):(chunk_size * i + chunk_size-1)]
        master_list.append(mini_list)
    return (master_list)


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
school_name_chunks = chunk_my_list(school_name_list, 100)


# %%


# %%
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# %%
system_prompt = """
you are a helpful chatbot that is great at finding contact details. You will be given a list of UK-based IELTS schools. Find the contact details of each and return in JSON format. Return only for each item the "phone", "email" and "address" fields populated for that school. Return the JSON only.
"""

# %%
prompt_template = prompts.ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("user", "{school_name_list}")
    ]
)

prompt = prompt_template.invoke({"school_name_list": school_names})


# %%
response = model.invoke(prompt)

# %%
# process the response


json_response = llm_response_to_json(response.content)

df = pd.DataFrame(json.loads(
    json_response), index=school_names).reset_index(names="school")


# %%
school_data


# %%
