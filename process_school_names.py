# %%
from openai import OpenAI
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
    """A class for posting to and pulling from LLM"""

    def __init__(self,
                 model="gemini-2.0-flash",
                 model_provider="google_genai",
                 system_prompt="""
you are a helpful chatbot that is great at finding contact details. You will be given a list of UK-based IELTS schools. Find the contact details of each and return in JSON format. Return only for each item the "phone", "email" and "address" fields populated for that school. Return the JSON only. The JSON object MUST have the same number of items as the number of schools you were given as input.
"""):

        self.model = model
        self.model_provider = model_provider
        self.system_prompt = system_prompt

    def llm_response_to_json(self, content):
        output = content
        output = output.removeprefix("```json")
        output = output.removesuffix("```")
        return output

    def collect_single_list(self, items_list):
        model = init_chat_model(
            model=self.model,
            model_provider=self.model_provider
        )
        prompt_template = prompts.ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("user", "{list_of_names}")
            ]
        )
        prompt = prompt_template.invoke({"list_of_names": items_list})
        response = model.invoke(prompt)
        json_response = self.llm_response_to_json(response.content)
        list_response = json.loads(json_response)
        df = pd.DataFrame(list_response, index=items_list).reset_index(
            names="item_name")
        return df

    def collect_chunked_list(self, items_list):
        output_dfs = []
        i = 0
        for chunk in items_list:
            i = i + 1
            print(f"collecting for chunk {i} of {len(items_list)}")
            df = self.collect_single_list(chunk)
            output_dfs.append(df)

        details_complete = pd.concat(
            output_dfs).sort_values("item_name").set_index("item_name")

        return details_complete


# %%

# ---------------- run through using gemini ---------------

google_llm_class = llm_details_puller(
    model="gemini-2.0-flash",
    model_provider="google_genai",
)
school_details_gemini = google_llm_class.collect_chunked_list(
    school_name_chunks)

school_details_gemini.to_csv("IELTS Details - Gemini.csv")

# %%

# ---------------- run through using chatGPT ---------------

openai_llm_class = llm_details_puller(
    model="o4-mini-2025-04-16",
    model_provider="openai",
)
school_details_openai = openai_llm_class.collect_chunked_list(
    school_name_chunks)

school_details_openai.to_csv("IELTS Details - ChatGPT.csv")
