import numpy as np
import yaml
from numpy.linalg import norm
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch
from pinecone import Pinecone
from pinecone import ServerlessSpec
from transformers import BitsAndBytesConfig
from huggingface_hub import InferenceClient
import streamlit as st



pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index_name = "chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index("chatbot")

with open("new_array.json", "r", encoding="utf-8") as f:
    loaded_array = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)



model_emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
model_emb = model_emb.to(device)

wiki_emb = model_emb.encode(loaded_array)

data_to_upsert = []

for i, (embedding, text) in enumerate(zip(wiki_emb, loaded_array)):
    data_to_upsert.append({
        "id": f"vec_{i}",          # Unique ID for each vector
        "values": embedding.tolist(), # Convert numpy array to list
        "metadata": {"text": text}    # Storing original text to retrieve it later
    })

# 2. Upsert to the index in batches (Pinecone recommends batches of ~100)
if "vectors_uploaded" not in st.session_state:
    index.upsert(vectors=data_to_upsert)
    st.session_state["vectors_uploaded"] = True

#print(f"Successfully uploaded {len(data_to_upsert)} vectors to Pinecone.")

with open("new_prompt.yml", "r", encoding="utf-8") as f:
    prompt_config = yaml.safe_load(f)

system_prompt_template = prompt_config["system_prompt"]


user_input = ""
dialog_queries = [] 

class QwenChatbot:
    def __init__(self):
        self.client = InferenceClient(
            model="Qwen/Qwen2.5-7B-Instruct",   # better supported
            token=st.secrets["HF_TOKEN"],
        )
        self.history = []

    def generate_response(self, user_input, system_prompt):

        messages = [
            {"role": "system", "content": system_prompt}
        ] + self.history + [
            {"role": "user", "content": user_input}
        ]

        response = self.client.chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )

        answer = response.choices[0].message["content"]

        # Save history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})

        return answer

st.title("Chatbot")

# Load chatbot once
@st.cache_resource
def load_chatbot():
    return QwenChatbot()

chatbot = load_chatbot()

# Chat input
user_input = st.chat_input("Enter message")

if user_input:

    # Embed query
    query_emb = model_emb.encode([user_input])
    query_vector = query_emb[0].tolist()

    # Retrieve from Pinecone
    results = index.query(
        vector=query_vector,
        top_k=1,
        include_metadata=True
    )

    if results["matches"]:
        retrieved_context = results["matches"][0]["metadata"]["text"]
    else:
        retrieved_context = "No relevant context found."

    # Build prompt
    prompt = system_prompt_template.format(
        context=retrieved_context
    )

    # Generate response
    bot_response = chatbot.generate_response(user_input, prompt)

    # Display
    st.write("**Answer:**")
    st.write(bot_response)




    




