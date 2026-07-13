from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

print("1. Starting...")

load_dotenv()

print("2. Environment loaded")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

print("3. LLM created")

model = ChatHuggingFace(llm=llm)

print("4. Chat model created")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain")
]

print("5. Invoking model...")

result = model.invoke(messages)

print("6. Model responded")

messages.append(AIMessage(content=result.content))

print(messages)