from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
import google.generativeai as genai
import os

api = os.environ.get("GOOGLE_API_KEY")

genai.configure(api_key=api)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key = 'AIzaSyDAnpnC2AX3UYd3fXwIe1T1V1wLOsDEXbg'
)

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="HFT-DEMO", 
    embedding=embeddings,
)

seach_query = "what is HFT trading"

# message = []

while True:

    user_query = input(">> ")

    # message.append({"user" : user_query})

    seach_req = retriver.similarity_search(
        query=user_query,
    )

    # print("relevent change" , seach_req)

    system_prompt = f'''
                Based on the user's question, determine which book pages is most relevant. 

                The available book content is as follows:
                {seach_req}
            
                Respond with ONLY the Book Content exactly as shown, nothing else.

                give pages number with answer.
    '''
    
    model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-preview-05-20",
                system_instruction=system_prompt,
                generation_config={"response_mime_type": "application/json"}
            )

    response = model.generate_content(user_query)


    answer = response.text

    print("🤖" ,answer)

    # message.append({"AI" : answer})