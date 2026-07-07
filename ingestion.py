import argparse
import json
import os
import pickle
import re
import urllib.error
import urllib.request

import faiss
import numpy as np

DEFAULT_EMBEDDING_MODEL = 'nomic-embed-text'


def _extract_pdf_text(file_path):
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - fallback for environments without pypdf
        try:
            from PyPDF2 import PdfReader
        except ImportError:  # pragma: no cover - fallback for environments without any PDF reader
            raise RuntimeError('PDF parsing requires pypdf or PyPDF2 to be installed.') from None

    reader = PdfReader(file_path)
    pages = [page.extract_text() or '' for page in reader.pages]
    return '\n\n'.join(page for page in pages if page).strip()


def read_text(file_path):
    if os.path.isdir(file_path):
        content_parts = []
        for root, _, files in os.walk(file_path):
            for fname in files:
                if fname.startswith('.'):
                    continue
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                try:
                    if ext == '.pdf':
                        content_parts.append(_extract_pdf_text(fpath))
                    elif ext in {'.txt', '.md', '.rst', '.json', '.csv', '.html', '.xml'}:
                        with open(fpath, 'r', encoding='utf-8') as handle:
                            content_parts.append(handle.read())
                except Exception:
                    continue
        return '\n\n'.join(part for part in content_parts if part)

    if os.path.splitext(file_path)[1].lower() == '.pdf':
        return _extract_pdf_text(file_path)

    with open(file_path, 'r', encoding='utf-8') as handle:
        return handle.read()


def _chunk_text(text, chunk_size=300, overlap=60):
    paragraphs = [paragraph.strip() for paragraph in re.split(r'\n\s*\n', text) if paragraph.strip()]
    if not paragraphs:
        return []

    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(paragraph):
                break
            start = max(start + chunk_size - overlap, end)

    return chunks


def parse_document(text):
    chunks = _chunk_text(text)
    if not chunks:
        return []

    records = []
    for index, chunk in enumerate(chunks, start=1):
        first_line = next((line.strip() for line in chunk.splitlines() if line.strip()), '').strip()
        title = re.sub(r'^#+\s*', '', first_line)[:80] or f'Chunk {index}'
        records.append({
            'id': index,
            'title': title,
            'text': chunk,
            'raw': chunk,
        })
    return records


def embed_texts(texts, model_name=DEFAULT_EMBEDDING_MODEL):
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    timeout = int(os.getenv('OLLAMA_TIMEOUT', '600'))
    payload = {'model': model_name, 'input': list(texts)}
    req = urllib.request.Request(
        f'{host}/api/embed',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
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


def ingest_file(file_path, model_name=DEFAULT_EMBEDDING_MODEL):
    content = read_text(file_path)
    records = parse_document(content)
    if not records:
        records = [{
            'id': 1,
            'title': 'Document',
            'text': content or 'No content extracted.',
            'raw': content or 'No content extracted.',
        }]

    texts = [record['text'] for record in records]
    embeddings = embed_texts(texts, model_name=model_name)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, 'faiss_index.faiss')
    np.save('embeddings.npy', embeddings)
    with open('chunks.pkl', 'wb') as handle:
        pickle.dump(records, handle)
    with open('model_name.txt', 'w', encoding='utf-8') as handle:
        handle.write(model_name)

    return np.arange(len(records))


def main():
    parser = argparse.ArgumentParser(description='Ingest a document into an embedding-based retrieval index')
    parser.add_argument('file_path', nargs='?', default='.', help='Path to the source file or directory')
    parser.add_argument('--model', default=DEFAULT_EMBEDDING_MODEL, help='Ollama embedding model name')
    args = parser.parse_args()

    result = ingest_file(args.file_path, model_name=args.model)
    print(result)


if __name__ == '__main__':
    main()