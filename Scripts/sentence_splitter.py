# Splitting sentences of CORD-19 with Genia Sentence Splitter.

import os
import pandas as pd
import json
import csv
import os
from subprocess import call
from tqdm import tqdm

def get_abstract(path, full_if_empty=True):
    with open(path, "r") as f:
        x = json.load(f)
    paper_id = x["paper_id"]
    if "abstract" in x:
        abstract = "\n".join([el["text"] for el in x["abstract"]])
    else:
        abstract = ""
    with open("temp.txt", "w") as f:
        f.write(abstract)
        
    status = call("./geniass ../Scripts/temp.txt ../Scripts/temps.txt", cwd="../geniass/",shell=True)
    with open("temps.txt", "r") as f:
        sentences = f.read().splitlines()
    
    os.remove("temp.txt")
    os.remove("temps.txt")
    return paper_id, sentences

try:
    os.remove("temp.txt")
except OSError:
    pass

try:
    os.remove("temps.txt")
except OSError:
    pass



for source in ["pdf_json", "pmc_json"]:
    folder = os.path.join("../Data/Raw/document_parses/", source)
    total_list = {}
    for js in tqdm(os.listdir(folder)):
        paper_id, sentences = get_abstract(os.path.join(folder, js))
        total_list[paper_id] = sentences

    with open(os.path.join("../Data/Abstract_Splitted/", source+".json"), "w") as f:
        json.dump(total_list, f)


