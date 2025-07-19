
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import google.generativeai as genai

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

pdf_path = Path(__file__).parent /"PDF.pdf"

loader = PyPDFLoader(file_path=pdf_path)

doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100 
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key = 'AIzaSyDzyFEdafRrhZn-W63lcjqx42TbQ_1YiYw')
embeddings.embed_query("What's our Q1 revenue?")

split_text = text_splitter.split_documents(documents=doc)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="Demo-1",
#     embedding=embeddings,
# )

# vector_store.add_documents(documents=split_text)

print("Injection Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="Demo-1",
    embedding=embeddings,
)

userquery = "What is Performing a Lightweight Document Check and give one code " 

retriver_chunks = retriver.similarity_search(
    query=userquery,
)

System_prompt = f'''
    you are a teaching aide for a student who is learning about js.
    you responds base on the context of aveliable context.

    output must be in 250 words easy to understand format.

    context:
    {retriver_chunks}
'''

print(System_prompt)

genai.configure(api_key="AIzaSyDzyFEdafRrhZn-W63lcjqx42TbQ_1YiYw")


model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-05-20",
            system_instruction=System_prompt,
            generation_config={"response_mime_type": "application/json"}
        )

response = model.generate_content(userquery)

print(response.text)