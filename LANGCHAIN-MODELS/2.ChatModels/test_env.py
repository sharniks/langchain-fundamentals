from dotenv import load_dotenv, find_dotenv
import os

env_path = find_dotenv()

print("Found .env:", env_path)

loaded = load_dotenv()

print("Loaded:", loaded)

print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))
print("HF_TOKEN:", os.getenv("HF_TOKEN"))
print("HUGGINGFACEHUB_API_TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))
print("HUGGINGFACEHUB_ACCESS_TOKEN:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))