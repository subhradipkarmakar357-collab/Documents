import argparse
import os
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = 'all-MiniLM-L6-v2'


def read_text(file_path):
    if os.path.isdir(file_path):
        content_parts = []
        for root, _, files in os.walk(file_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as handle:
                        content_parts.append(handle.read())
                except Exception:
                    continue
        return '\n\n'.join(content_parts)

    with open(file_path, 'r', encoding='utf-8') as handle:
        return handle.read()


def parse_document(text):
    records = []
    blocks = re.split(r'\n(?=\d+\.\s+\*\*)', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        match = re.match(r'(\d+)\.\s+\*\*(.+?)\*\*\s+\(([^)]+)\)', block)
        if not match:
            continue

        number, title, year = match.groups()
        record = {
            'id': int(number),
            'title': title.strip(),
            'year': year.strip(),
            'description': '',
            'genre': '',
            'rating': '',
            'director': '',
            'actors': '',
            'release_date': '',
            'raw': block,
        }

        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('Description:'):
                record['description'] = line.split(':', 1)[1].strip()
            elif line.startswith('Genre:'):
                record['genre'] = line.split(':', 1)[1].strip()
            elif line.startswith('Rating:'):
                record['rating'] = line.split(':', 1)[1].strip()
            elif line.startswith('Director:'):
                record['director'] = line.split(':', 1)[1].strip()
            elif line.startswith('Actors:'):
                record['actors'] = line.split(':', 1)[1].strip()
            elif line.startswith('Release Date:'):
                record['release_date'] = line.split(':', 1)[1].strip()

        record['text'] = (
            f"Title: {record['title']} ({record['year']}). "
            f"Description: {record['description']}. "
            f"Genre: {record['genre']}. "
            f"Rating: {record['rating']}. "
            f"Director: {record['director']}. "
            f"Actors: {record['actors']}. "
            f"Release Date: {record['release_date']}."
        )
        records.append(record)

    if not records:
        for i, paragraph in enumerate(re.split(r'\n\s*\n', text)):
            paragraph = paragraph.strip()
            if paragraph:
                records.append({
                    'id': i + 1,
                    'title': f'Chunk {i + 1}',
                    'year': '',
                    'text': paragraph,
                    'raw': paragraph,
                })

    return records


def ingest_file(file_path, model_name=DEFAULT_MODEL):
    content = read_text(file_path)
    records = parse_document(content)
    texts = [record['text'] for record in records]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')

    index = faiss.IndexFlatIP(embeddings.shape[1])
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
    parser.add_argument('file_path', nargs='?', default='OscarMovies', help='Path to the source file or directory')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Sentence-transformers model name')
    args = parser.parse_args()

    result = ingest_file(args.file_path, model_name=args.model)
    print(result)


if __name__ == '__main__':
    main()