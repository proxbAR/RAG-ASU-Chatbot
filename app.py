import streamlit as st
from vertexai.language_models import TextGenerationModel
import vertexai
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# ---ENVIRONMENT--- #
load_dotenv()

# ---CONFIG--- #
PROJECT = os.getenv("PROJECT")
LOCATION = os.getenv("LOCATION")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH")
QDRANT_REST_URL=os.getenv("QDRANT_URL")
QDRANT_REST_KEY=os.getenv("QDRANT_API_KEY")

# ---SETUP--- # 
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
vertexai.init(project=PROJECT, location=LOCATION, credentials=creds)
model = TextGenerationModel.from_pretrained("text-bison")

qdrant = QdrantClient(
    url=QDRANT_REST_URL,
    api_key=QDRANT_REST_KEY,
)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---STREAMLIT--- #
st.set_page_config(page_title="ASUCHAT", page_icon="🎓")
st.title("ASUChat: Ask me about ASU!")

user_question = st.text_input("Ask me anything about ASU (and get a James Bond style reply):")

if st.button("Ask") and user_question:
    query_vec = embedder.encode(user_question)
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_vec, limit=3)

    if results and results[0].score > 0.7:
        context = "\n".join([r.payload['text'] for r in results])
    else:
        context = ""

    prompt = f"""
You are an AI assistant for ASU, and I want you to reply like James Bond for all queries. You are trained on the following FAQ knowledge base:

{context}

User Question: {user_question}
Answer:
"""

    response = model.predict(prompt=prompt, temperature=0.3, max_output_tokens=256)
    st.success(response.text.strip())
