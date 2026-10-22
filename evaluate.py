
import chromadb
from sentence_transformers import SentenceTransformer

# Load the same model and database used by main.py
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("company_documents")

# Questions and their expected relevant chunks
test_cases = [
    {
        "question": "How many vacation days do employees receive?",
        "expected_id": "leave_policy_0",
    },
    {
        "question": "How early must leave be requested?",
        "expected_id": "leave_policy_1",
    },
    {
        "question": "Can I work from home?",
        "expected_id": "remote_work_0",
    },
    {
        "question": "How often should I update my password?",
        "expected_id": "security_policy_0",
    },
    {
        "question": "What is the minimum password length?",
        "expected_id": "security_policy_1",
    },
]

k = min(3, collection.count())
hits = 0

print(f"\nEvaluating retrieval with Hit Rate@{k}\n")

for test in test_cases:
    question = test["question"]
    expected_id = test["expected_id"]

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    retrieved_ids = results["ids"][0]
    hit = expected_id in retrieved_ids

    if hit:
        hits += 1

    print(f"Question: {question}")
    print(f"Expected chunk: {expected_id}")
    print(f"Retrieved chunks: {retrieved_ids}")
    print(f"Result: {'HIT' if hit else 'MISS'}")
    print("-" * 50)

hit_rate = hits / len(test_cases)

print(f"\nHits: {hits}/{len(test_cases)}")
print(f"Hit Rate@{k}: {hit_rate:.2%}")
