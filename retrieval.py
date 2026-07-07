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

DEFAULT_EMBEDDING_MODEL = 'nomic-embed-text'
DEFAULT_LLM_MODEL = 'llama3.2:latest'


def load_artifacts():
    try:
        with open('chunks.pkl', 'rb') as handle:
            chunks = pickle.load(handle)
        with open('model_name.txt', 'r', encoding='utf-8') as handle:
            model_name = handle.read().strip() or DEFAULT_EMBEDDING_MODEL
        index = faiss.read_index('faiss_index.faiss')
    except Exception as exc:
        raise FileNotFoundError('Required artifacts not found. Run ingestion first.') from exc

    return model_name, chunks, index


def format_chunk(chunk):
    if isinstance(chunk, dict):
        title = chunk.get('title', '').strip()
        body = chunk.get('text', '').strip()
        if title and body and body != title:
            return f'{title}\n{body}'
        return title or body
    return str(chunk)


def embed_texts(texts, model_name=DEFAULT_EMBEDDING_MODEL):
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    payload = {'model': model_name, 'input': list(texts)}
    req = urllib.request.Request(
        f'{host}/api/embed',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
        raise RuntimeError(f'Unable to embed text with Ollama: {exc}') from exc

    embeddings = body.get('embeddings')
    if not embeddings:
        raise RuntimeError('Ollama did not return any embeddings.')

    array = np.array(embeddings, dtype='float32')
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def retrieve_file(query, k=5):
    model_name, chunks, index = load_artifacts()
    if not chunks:
        return []

    qvec = embed_texts([query], model_name=model_name)
    if qvec.ndim == 1:
        qvec = qvec.reshape(1, -1)

    search_k = len(chunks)
    _, indices = index.search(qvec, search_k)
    results = []
    for idx in indices[0]:
        if 0 <= int(idx) < len(chunks):
            results.append((float(idx), chunks[int(idx)]))

    return results[:k]


def _synthesize_with_ollama(question, chunks):
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    model_name = os.getenv('OLLAMA_LLM_MODEL', DEFAULT_LLM_MODEL)

    if not chunks:
        return 'No information in the provided context.'

    context_text = '\n\n'.join(
        f"Chunk {index + 1}: {chunk.get('text', '').strip()}"
        for index, chunk in enumerate(chunks[:8])
    )

    prompt = (
        f"You are a helpful assistant. Use only the information in the provided context.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer concisely and do not invent facts."
    )

    payload = {
        'model': model_name,
        'prompt': prompt,
        'stream': False,
        'options': {'temperature': 0.0},
    }

    req = urllib.request.Request(
        f'{host}/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode('utf-8'))
            return body.get('response', '').strip() or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, ConnectionRefusedError, socket.timeout, OSError):
        return None


def answer_query(query, k=5):
    _, chunks, _ = load_artifacts()
    results = retrieve_file(query, k=k)
    ranked_chunks = [chunk for _, chunk in results]

    start_time = time.perf_counter()
    text = _synthesize_with_ollama(query, ranked_chunks)
    latency = time.perf_counter() - start_time

    if text:
        return f'{text}\n\nLLM latency: {latency:.3f} seconds'

    return 'Ollama did not return a response. Check that the Ollama server is running and the requested model is available.'


def main():
    parser = argparse.ArgumentParser(description='Query the embedding-based FAISS index of documents')
    parser.add_argument('query', nargs='*', help='Query text (if omitted, enters interactive prompt)')
    parser.add_argument('-k', '--k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--answer', action='store_true', help='Return a synthesized answer using the local Ollama model instead of raw chunks')
    args = parser.parse_args()

    if args.query:
        q = ' '.join(args.query)
        try:
            if args.answer:
                ans = answer_query(q, k=args.k)
                print(ans)
                return
            results = retrieve_file(q, k=args.k)
        except FileNotFoundError as exc:
            print(exc)
            return

        for _, chunk in results:
            print(format_chunk(chunk))
            print('---')
    else:
        try:
            while True:
                q = input('Enter query (Ctrl-D to exit): ').strip()
                if not q:
                    continue
                try:
                    if args.answer:
                        ans = answer_query(q, k=args.k)
                        print(ans)
                        continue
                    results = retrieve_file(q, k=args.k)
                except FileNotFoundError as exc:
                    print(exc)
                    return

                for _, chunk in results:
                    print(format_chunk(chunk))
                    print('---')
        except EOFError:
            print('\nExiting.')


if __name__ == '__main__':
    main()