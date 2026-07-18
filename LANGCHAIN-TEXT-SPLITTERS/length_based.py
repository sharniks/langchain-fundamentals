from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

current_dir = Path(__file__).parent
file_path = current_dir / 'dl-curriculum.pdf'

loader = PyPDFLoader(str(file_path))

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print(result[0].page_content)
