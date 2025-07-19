from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

pdf_path = Path(__file__).parent /"PDF.pdf"

loader = PyPDFLoader(file_path=pdf_path)

doc = loader.load()

# print(doc[59])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents=doc)

# print(len(split_docs))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key = 'AIzaSyDAnpnC2AX3UYd3fXwIe1T1V1wLOsDEXbg'
)
 
vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="HFT-DEMO",
    embedding=embeddings,
    force_recreate=True
)

vector_store.add_documents(documents=split_docs)

print('Injection Done')