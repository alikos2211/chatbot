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

pc = Pinecone(api_key="pcsk_42FVuS_8pBZ6qCiTdPnoXfrxfhK59QJYmWS2nW4UVvDiqcbALMBrwRfnzDQm3Qfb2DARDQ")
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
index.upsert(vectors=data_to_upsert)

#print(f"Successfully uploaded {len(data_to_upsert)} vectors to Pinecone.")

with open("new_prompt.yml", "r", encoding="utf-8") as f:
    prompt_config = yaml.safe_load(f)

system_prompt_template = prompt_config["system_prompt"]


user_input = ""
dialog_queries = [] 

class QwenChatbot:
    def __init__(self, model= "Qwen/Qwen3-8B"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=quantization_config,
            device_map="auto" 
        )
        self.model = self.model.to(device)
        self.history = []
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def generate_response(self, user_input):
        messages = self.history + [
            {"role": "user", "content": user_input},
            
                              ]
        

        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(texts, prompt, return_tensors="pt").to(device)
        response_ids = self.model.generate(**inputs, 
                                           max_new_tokens=80, 
                                           do_sample=True, 
                                           temperature=0.7, 
                                           top_p=0.9)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

if __name__ == "__main__":
    chatbot = QwenChatbot()

while (user_input != "Stop"):
    user_input = input("Enter message: ")

    query_emb = model_emb.encode(user_input)
    query_vector = query_emb.tolist()

    # Query Pinecone (similarity)
    results = index.query(
    vector=query_vector,
    top_k=1,
    include_metadata=True
    )
    # Get the context from the top result
    if results['matches']:
        retrieved_context = results['matches'][0]['metadata']['text']
    else:
        retrieved_context = "No relevant context found."

    prompt = system_prompt_template.format(
        context=retrieved_context,
    )
 

    bot_response = chatbot.generate_response(user_input)
 
    print("Answer:", bot_response)
    #print(chatbot.history)


    




