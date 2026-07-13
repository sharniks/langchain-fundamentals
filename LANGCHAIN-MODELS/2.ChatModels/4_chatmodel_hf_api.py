from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# you can use 'meta-llama/Llama-3.1-8B-Instruct' as well
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

result = model.invoke(
    "What is the capital of US, also tell 2 sentence about it")

print(result.content)
