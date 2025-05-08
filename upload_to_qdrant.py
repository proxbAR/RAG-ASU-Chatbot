from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
from dotenv import load_dotenv

# ---ENVIRONMENT--- #
load_dotenv()

QDRANT_REST_URL=os.getenv("QDRANT_URL")
QDRANT_REST_KEY=os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(
    url=QDRANT_REST_URL,
    api_key=QDRANT_REST_KEY
)

# Create collection if it doesn’t exist
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

with open("asu.txt") as f:
    blocks = f.read().split("\n\n")

points = []
for idx, block in enumerate(blocks):
    if block.strip():
        try:
            question, answer = block.split("\n", 1)
            vector = embedder.encode(question)
            points.append(PointStruct(id=idx, vector=vector, payload={"question": question.strip(), "text": answer.strip()}))
        except ValueError:
            print(f"Skipping block {idx} — invalid format")

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"Uploaded {len(points)} QA pairs to Qdrant")