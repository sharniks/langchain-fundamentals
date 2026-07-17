from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pathlib import Path

current_dir = Path(__file__).parent

loader = DirectoryLoader(
    path=current_dir/'books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(docs[325].page_content)
print(docs[325].metadata)
