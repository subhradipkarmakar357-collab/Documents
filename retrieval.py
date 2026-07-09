
import argparse
import json
import os
import pickle
import socket
import time
import urllib.error
import urllib.request

import faiss
import numpy as np
from sentence_transformers import CrossEncoder
from rag_evaluation import evaluate_pipeline

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

OLLAMA_LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:latest"
)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.0"))

print("Loading reranker...")
RERANKER = CrossEncoder(
    os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
)
print("Reranker loaded.")

def load_artifacts():
    with open("chunks.pkl","rb") as f:
        chunks=pickle.load(f)
    with open("model_name.txt","r",encoding="utf-8") as f:
        model=f.read().strip() or OLLAMA_EMBEDDING_MODEL
    index=faiss.read_index("faiss_index.faiss")
    return model,chunks,index

def embed_texts(texts, model_name=OLLAMA_EMBEDDING_MODEL):
    host=os.getenv("OLLAMA_HOST","http://localhost:11434")
    payload={"model":model_name,"input":list(texts)}
    req=urllib.request.Request(
        f"{host}/api/embed",
        data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req,timeout=60) as resp:
        body=json.loads(resp.read().decode())
    emb=np.array(body["embeddings"],dtype="float32")
    if emb.ndim==1:
        emb=emb.reshape(1,-1)
    return emb

def retrieve_file(query, k=20):
    model, chunks, index = load_artifacts()
    qvec = embed_texts([query], model)
    distances, indices = index.search(qvec, len(chunks))
    results = []
    print(f"Distances: {distances[0]}")
    for d, i in zip(distances[0], indices[0]):
        if 0 <= i < len(chunks):
            # Skip chunks whose L2 distance is too high
            if d > SIMILARITY_THRESHOLD:
                continue
            results.append((float(d), chunks[int(i)]))

    return results[:k]

def rerank_results(query, retrieved_results, top_n=os.getenv("TOP_K_RERANK", 5)):

    pairs = [
        (query, chunk["text"])
        for _, chunk in retrieved_results
    ]

    scores = RERANKER.predict(pairs)

    ranked = sorted(
        zip(scores, retrieved_results),
        key=lambda x: x[0],
        reverse=True,
    )

    top_results = [result for _, result in ranked[:top_n]]
    top_scores = [float(score) for score, _ in ranked[:top_n]]

    return top_results, top_scores

def synthesize(question,chunks):
    host=os.getenv("OLLAMA_HOST","http://localhost:11434")
    model=os.getenv("OLLAMA_LLM_MODEL",OLLAMA_LLM_MODEL)
    context="\n\n-----\n\n".join(c["text"] for c in chunks)
    prompt=f"""
You are an expert assistant.
Use ONLY the reference material.
Never mention chunks, context, retrieved passages or document sections.
Combine information from multiple passages.
Provide a detailed answer with:
Overview
Detailed Explanation
Key Points
Conclusion

Reference Material:
{context}

Question:
{question}

Answer:
"""
    payload={"model":model,"prompt":prompt,"stream":False,"options":{"temperature":0}}
    req=urllib.request.Request(f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=int(os.getenv("OLLAMA_TIMEOUT", "600"))) as resp:
        return json.loads(resp.read().decode())["response"].strip()

def answer_query(query):
    # -----------------------------
    # Retrieval
    # -----------------------------
    retrieval_start = time.perf_counter()

    retrieved = retrieve_file(query, k=20)

    retrieval_time = time.perf_counter() - retrieval_start

    # -----------------------------
    # Reranking
    # -----------------------------
    rerank_start = time.perf_counter()

    reranked, reranker_scores = rerank_results(
    query,
    retrieved,
    top_n=5,
    )

    rerank_time = time.perf_counter() - rerank_start

    chunks = [chunk for _, chunk in reranked]

    # -----------------------------
    # LLM Generation
    # -----------------------------
    generation_start = time.perf_counter()

    ans = synthesize(query, chunks)

    generation_time = time.perf_counter() - generation_start

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    metrics = evaluate_pipeline(
        retrieved_count=len(retrieved),
        reranked_scores=reranker_scores,
        retrieval_time=retrieval_time,
        rerank_time=rerank_time,
        generation_time=generation_time,
    )

    # -----------------------------
    # Final Response
    # -----------------------------
    return (
        f"{ans}\n\n"
        f"{metrics.format_report()}"
    )

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("query",nargs="*")
    parser.add_argument("--answer",action="store_true")
    args=parser.parse_args()
    q=" ".join(args.query)
    if args.answer:
        print(answer_query(q))
    else:
        res=rerank_results(q,retrieve_file(q,20),5)
        for _,c in res:
            print(c["text"])
            print("-"*80)

if __name__=="__main__":
    main()
