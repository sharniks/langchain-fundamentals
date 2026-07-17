from langchain_community.document_loaders import CSVLoader
from pathlib import Path

current_dir = Path(__file__).parent

loader = CSVLoader(file_path=current_dir/'Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1])
