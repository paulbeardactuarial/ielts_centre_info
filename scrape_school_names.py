# %%
import re
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import json
import string
import requests


# %%

base_url = "https://www.britishcouncil.org/education/accreditation/centres/"
search_urls = [f"{base_url}{letter}" for letter in list(
    string.ascii_lowercase)]

# %%
test = search_urls[0]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://google.com"
})

response = session.get(test, timeout=10)

# %%
