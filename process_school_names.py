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
import random

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


def shuffle_with_seed(input_list, seed=42):
    rng = random.Random(seed)
    shuffled = input_list[:]
    rng.shuffle(shuffled)
    return shuffled


# %%
data_raw = pd.read_csv(
    "raw_school_names.txt",
    delimiter="/n",
    header=None,
    names=["school_raw"]
)

# %%
max_chunk_size = 50

data_raw["school_clean"] = data_raw["school_raw"].str.replace(
    r'Full report \d{4}\s\(PDF \d+KB\)', "", regex=True, case=False)
data_raw["school_clean"] = data_raw["school_clean"].str.extract(r"^(.*?),")
school_name_list = list(data_raw["school_clean"].dropna().unique())
# shuffle the order of schools to random...
# ...this step is important to reduce chance of LLM taking shortcuts
# ...and not providing information for similar (duplicate) school
# ...we don't want LLM to change structure/size, even if there are dupes
school_name_list = shuffle_with_seed(school_name_list, seed=23)
school_name_chunks = chunk_my_list(school_name_list, max_chunk_size)


# %%
system_prompt = """
you are a helpful chatbot that is great at finding contact details. You will be given a list of UK-based IELTS schools. Find the contact details of each and return in JSON format. Return only for each item the "phone", "email" and "address" fields populated for that school. Return the JSON only. The JSON object MUST have the same number of items as the number of schools you were given as input.
"""


# %%
class llm_details_puller:


def llm_give_me_the_details(
        system_prompt,
        list_chunks,
        model="gemini-2.0-flash",
        model_provider="google_genai",
):
    output_dfs = []

    for chunk in list_chunks:
        model = init_chat_model(
            model,
            model_provider=model_provider
        )
        prompt_template = prompts.ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("user", "{list_of_names}")
            ]
        )
        prompt = prompt_template.invoke({"list_of_names": chunk})
        response = model.invoke(prompt)
        json_response = llm_response_to_json(response.content)
        list_response = json.loads(json_response)
        df = pd.DataFrame(list_response, index=chunk).reset_index(
            names="name")
        output_dfs.append(df)

    details_complete = pd.concat(
        output_dfs).sort_values("name").set_index("name")

    return (details_complete)


# %%
school_details_gemini = llm_give_me_the_details(
    system_prompt=system_prompt,
    list_chunks=school_name_chunks,
    model="gemini-2.0-flash",
    model_provider="google_genai",
)

# %%
school_details_openai = llm_give_me_the_details(
    system_prompt=system_prompt,
    list_chunks=school_name_chunks,
    model="gpt-4o-mini",
    model_provider="openai",
)

# %%
