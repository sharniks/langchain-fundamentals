import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

# Dynamically finds the directory of this script file
current_dir = Path(__file__).parent
file_path = current_dir / 'cricket.txt'

# print("Python is looking for your file here:", os.getcwd())

loader = TextLoader(str(file_path), encoding='utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))

##print(docs[0])

chain = prompt | model | parser

print(chain.invoke({'poem': docs[0].page_content}))
