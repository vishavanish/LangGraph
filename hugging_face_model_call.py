from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv() 

client = InferenceClient(
    model="Qwen/Qwen2.5-72B-Instruct", # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    token=os.getenv("HUGGING_FACE_ACCESS_TOKEN")
)

response = client.chat_completion(
    messages=[
        {"role": "user", "content": "Hello! Who are you?"}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)