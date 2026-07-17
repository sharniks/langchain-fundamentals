from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

current_dir = Path(__file__).parent
file_path = current_dir / 'dl-curriculum.pdf'

loader = PyPDFLoader(str(file_path))

docs = loader.load()

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)
