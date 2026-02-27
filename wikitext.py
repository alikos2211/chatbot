import numpy as np
import json
from datasets import load_dataset

ds = load_dataset("Laz4rz/wikipedia_science_chunked_small_rag_512")

full_array = ds["train"]['text']
new_array = full_array[0:10]

with open("new_array.json", "w", encoding="utf-8") as f:
    json.dump(new_array, f, ensure_ascii=False, indent=2)