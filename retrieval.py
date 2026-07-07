import argparse
import json
import os
import pickle
import re
import socket
import time
import urllib.error
import urllib.request

import faiss
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = 'all-MiniLM-L6-v2'


def load_artifacts():
    try:
        with open('chunks.pkl', 'rb') as handle:
            chunks = pickle.load(handle)
        with open('model_name.txt', 'r', encoding='utf-8') as handle:
            model_name = handle.read().strip() or DEFAULT_MODEL
        index = faiss.read_index('faiss_index.faiss')
    except Exception as exc:
        raise FileNotFoundError('Required artifacts not found. Run ingestion first.') from exc

    model = SentenceTransformer(model_name)
    return model, chunks, index


def format_chunk(chunk):
    if isinstance(chunk, dict):
        pieces = []
        if chunk.get('title'):
            pieces.append(chunk['title'])
        if chunk.get('year'):
            pieces.append(f"({chunk['year']})")
        header = ' '.join(pieces)
        body = chunk.get('text', '')
        if header and body and str(body).strip() != str(header).strip():
            return f"{header}\n{body}"
        return header or body
    return str(chunk)


def _extract_year(query):
    years = re.findall(r'\b(?:19|20)\d{2}\b', query)
    return years[0] if years else None


def _extract_hyphenated_year(query):
    match = re.search(r'\b(19\d{2}|20\d{2})\s*-\s*(\d{2})\b', query, re.IGNORECASE)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def _extract_year_range(query):
    # patterns like 'between 1946 and 1950', 'from 1946 to 1950', '1946-47'
    match = re.search(r'\bbetween\s+(19\d{2}|20\d{2})\s+and\s+(19\d{2}|20\d{2})\b', query, re.IGNORECASE)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (min(start, end), max(start, end))

    match = re.search(r'\bfrom\s+(19\d{2}|20\d{2})\s+(?:to|until|through)\s+(19\d{2}|20\d{2})\b', query, re.IGNORECASE)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (min(start, end), max(start, end))

    match = re.search(r'\b(19\d{2}|20\d{2})\s*-\s*(\d{2})\b', query, re.IGNORECASE)
    if match:
        start = int(match.group(1))
        end = int(f"{match.group(1)[:2]}{match.group(2)}")
        return (min(start, end), max(start, end))

    return None


def _extract_director(query):
    patterns = [
        r'\b(?:directed by|films? by|movies? by)\s+([A-Za-z .\'\-]+)',
        r'\b([A-Za-z][A-Za-z .\'\-]+?)\s+(?:films|movies)\b',
        r'\bdirector\s+([A-Za-z .\'\-]+)',
    ]

    stopword_re = re.compile(r'\b(?:how|what|which|list|give|show|many|all)\b', re.IGNORECASE)

    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # ignore captures that are clearly part of the question ("how many", etc.)
            if stopword_re.search(candidate):
                continue
            return candidate

    return None


def _extract_actors(query):
    match = re.search(r'\b(?:starring|stars|starred by|starred|actors?)\s+([A-Za-z ,.\'-]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_genre(query):
    match = re.search(r'\bgenre\s+([A-Za-z\s-]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r'\b(?:romantic comedy|romantic drama|historical drama|psychological thriller|musical drama|war drama|historical romance|biographical drama|biographical musical|musical romantic comedy)\b', query, re.IGNORECASE)
    if match:
        return match.group(0).strip()

    return None


def _extract_release_date(query):
    match = re.search(r'\breleased(?: on| in)?\s+([A-Za-z0-9, ]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r'\brelease date\s+([A-Za-z0-9, ]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def _extract_rating(query):
    match = re.search(r'\brating\s+([A-Za-z0-9 /-]+)', query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_title(query):
    match = re.search(r'"([^"]+)"|\b(?:movie|film)\s+called\s+([A-Za-z0-9 \-\'\,]+)', query, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2)
    return None


def _extract_distinct_field(query):
    query_lower = query.lower()
    # require explicit list-style intent (avoid matching phrases like
    # "mention the movie name and genre")
    if not re.search(r'\b(?:distinct|unique|all|list|show|give|name the|names of|which are|what are)\b', query_lower):
        return None

    field_map = {
        'genre': 'genre',
        'genres': 'genre',
        'director': 'director',
        'directors': 'director',
        'actor': 'actors',
        'actors': 'actors',
        'year': 'year',
        'years': 'year',
        'release date': 'release_date',
        'release dates': 'release_date',
        'rating': 'rating',
        'ratings': 'rating',
        'title': 'title',
        'titles': 'title',
    }

    for field_name, field_value in field_map.items():
        if re.search(rf'\b{re.escape(field_name)}\b', query_lower):
            return field_value

    return None


def _normalize_text(value):
    return re.sub(r'[^a-z0-9]+', ' ', str(value).lower()).strip()


def _extract_requested_fields(query):
    """Return a list of output fields the user explicitly requested in the query.
    Examples: 'mention the movie name and genre' -> ['title','genre']
    """
    q = query.lower()
    field_map = {
        'title': ['title', 'name', 'movie name', 'movie', 'film name'],
        'genre': ['genre', 'genres'],
        'director': ['director', 'directed by'],
        'actors': ['actor', 'actors', 'starring', 'stars'],
        'year': ['year', 'years', 'released', 'release date'],
        'rating': ['rating'],
    }

    requested = []
    for key, synonyms in field_map.items():
        for syn in synonyms:
            if re.search(rf'\b{re.escape(syn)}\b', q):
                requested.append(key)
                break

    return requested


def _format_chunk_fields(chunk, fields):
    parts = []
    for f in fields:
        if f == 'title':
            parts.append(chunk.get('title', ''))
        elif f == 'genre':
            parts.append(chunk.get('genre', ''))
        elif f == 'director':
            parts.append(chunk.get('director', ''))
        elif f == 'actors':
            parts.append(chunk.get('actors', ''))
        elif f == 'year':
            parts.append(chunk.get('year', ''))
        elif f == 'rating':
            parts.append(chunk.get('rating', ''))
    # join non-empty parts with ' — '
    parts = [p for p in parts if p]
    return ' — '.join(parts) if parts else format_chunk(chunk)


def _matches_text(query_value, actual_value):
    if not query_value or not actual_value:
        return False

    query_norm = _normalize_text(query_value)
    actual_norm = _normalize_text(actual_value)

    if query_norm in actual_norm or actual_norm in query_norm:
        return True

    query_tokens = set(query_norm.split())
    actual_tokens = set(actual_norm.split())
    if not query_tokens or not actual_tokens:
        return False

    return query_tokens.issubset(actual_tokens) or actual_tokens.issubset(query_tokens)


def _matches_director(query_director, chunk):
    if not query_director or not isinstance(chunk, dict):
        return False

    return _matches_text(query_director, chunk.get('director', ''))


def _matches_actors(query_actors, chunk):
    return _matches_text(query_actors, chunk.get('actors', ''))


def _matches_genre(query_genre, chunk):
    return _matches_text(query_genre, chunk.get('genre', ''))


def _matches_release_date(query_release_date, chunk):
    if not query_release_date or not isinstance(chunk, dict):
        return False

    if _matches_text(query_release_date, chunk.get('release_date', '')):
        return True

    query_year = _extract_year(query_release_date)
    return query_year and query_year in str(chunk.get('release_date', ''))


def _matches_year(query_year, chunk):
    if not query_year or not isinstance(chunk, dict):
        return False

    return query_year in str(chunk.get('year', '')) or query_year in str(chunk.get('release_date', ''))


def _matches_year_range(year_range, chunk):
    if not year_range or not isinstance(chunk, dict):
        return False

    text_fields = f"{chunk.get('year',' ')} {chunk.get('release_date',' ')}"
    years = []
    for year_text in re.findall(r'\b(?:19|20)\d{2}(?:-\d{2})?\b', text_fields):
        if '-' in year_text:
            start_text, _ = year_text.split('-', 1)
            years.append(int(start_text))
        else:
            years.append(int(year_text))

    if not years:
        return False

    start, end = year_range
    return any(start <= y <= end for y in years)


def _matches_title(query_title, chunk):
    if not query_title or not isinstance(chunk, dict):
        return False

    return _matches_text(query_title, chunk.get('title', ''))


def _matches_rating(query_rating, chunk):
    return _matches_text(query_rating, chunk.get('rating', ''))


def _matches_field(field, query_value, chunk):
    if not query_value or not isinstance(chunk, dict):
        return False

    if field == 'director':
        return _matches_director(query_value, chunk)
    if field == 'actors':
        return _matches_actors(query_value, chunk)
    if field == 'genre':
        return _matches_genre(query_value, chunk)
    if field == 'release_date':
        return _matches_release_date(query_value, chunk)
    if field == 'year':
        return _matches_year(query_value, chunk)
    if field == 'year_range':
        return _matches_year_range(query_value, chunk)
    if field == 'title':
        return _matches_title(query_value, chunk)
    if field == 'rating':
        return _matches_rating(query_value, chunk)
    return False


def _extract_query_filters(query):
    filters = {}
    hyphen_year = _extract_hyphenated_year(query)
    if hyphen_year:
        filters['year'] = hyphen_year
    else:
        year_range = _extract_year_range(query)
        if year_range:
            filters['year_range'] = year_range
        else:
            year = _extract_year(query)
            if year:
                filters['year'] = year

    director = _extract_director(query)
    if director:
        filters['director'] = director

    actors = _extract_actors(query)
    if actors:
        filters['actors'] = actors

    genre = _extract_genre(query)
    if genre:
        filters['genre'] = genre

    release_date = _extract_release_date(query)
    if release_date:
        filters['release_date'] = release_date

    rating = _extract_rating(query)
    if rating:
        filters['rating'] = rating

    title = _extract_title(query)
    if title:
        filters['title'] = title

    return filters


def _boost_score(chunk, query):
    if not isinstance(chunk, dict):
        return 0.0

    boost = 0.0
    query_lower = query.lower()
    chunk_years = re.findall(r'\b(?:19|20)\d{2}\b', str(chunk.get('year', '')))
    query_years = re.findall(r'\b(?:19|20)\d{2}\b', query)
    if chunk_years and any(year in query_years for year in query_years):
        boost += 0.35

    title = str(chunk.get('title', '')).lower()
    if title and title in query_lower:
        boost += 0.15

    director = str(chunk.get('director', '')).lower()
    if director and director in query_lower:
        boost += 0.25

    actors = str(chunk.get('actors', '')).lower()
    if actors and any(actor.strip().lower() in query_lower for actor in actors.split(',')):
        boost += 0.20

    genre = str(chunk.get('genre', '')).lower()
    if genre and genre in query_lower:
        boost += 0.15

    return boost


def _rank_filtered_chunks(chunks, search_results, filters):
    if not filters:
        return [(score, chunk) for score, chunk in search_results]

    matched_chunks = [
        chunk for chunk in chunks
        if all(_matches_field(field, value, chunk) for field, value in filters.items())
    ]
    if not matched_chunks:
        return [(score, chunk) for score, chunk in search_results]

    score_lookup = {id(chunk): score for score, chunk in search_results}
    ranked_chunks = sorted(
        matched_chunks,
        key=lambda chunk: (score_lookup.get(id(chunk), float('inf')), str(chunk.get('title', '')).lower()),
    )
    return [(score_lookup.get(id(chunk), 0.0), chunk) for chunk in ranked_chunks]


def retrieve_file(query, k=5):
    model, chunks, index = load_artifacts()

    distinct_field = _extract_distinct_field(query)
    if distinct_field:
        filters = _extract_query_filters(query)
        values = []
        seen = set()
        for chunk in chunks:
            if filters and not all(_matches_field(field, value, chunk) for field, value in filters.items()):
                continue
            value = chunk.get(distinct_field, '')
            if not value:
                continue
            normalized = _normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(str(value))

        if values:
            values = sorted(values, key=lambda item: item.lower())
            return [(0.0, {'title': value, 'text': value, 'kind': distinct_field}) for value in values]

    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    search_k = len(chunks)
    D, I = index.search(qvec, k=search_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= int(idx) < len(chunks):
            chunk = chunks[int(idx)]
            boosted_score = float(dist) + _boost_score(chunk, query)
            results.append((boosted_score, chunk))

    results.sort(key=lambda item: item[0], reverse=True)

    filters = _extract_query_filters(query)
    if filters:
        ranked = _rank_filtered_chunks(chunks, results, filters)
        if ranked:
            return ranked[:k]

    return results[:k]


def _build_structured_answer(question, chunks):
    if not chunks:
        return 'No information in the provided context.'

    normalized_question = question.lower()
    if 'how many' in normalized_question and 'between' in normalized_question:
        lines = [f"{len(chunks)} movies:"]
        for idx, chunk in enumerate(chunks, start=1):
            title = chunk.get('title', 'Unknown')
            genre = chunk.get('genre', '') or 'Unknown'
            year = chunk.get('year', '')
            if year:
                lines.append(f"{idx}. {title} — {genre} ({year})")
            else:
                lines.append(f"{idx}. {title} — {genre}")
        return '\n'.join(lines)

    return None


def _synthesize_with_ollama(question, chunks):
    host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    model_name = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')

    structured_answer = _build_structured_answer(question, chunks)
    if structured_answer:
        return structured_answer

    # Build a focused CONTEXT from the top chunks
    ctx_parts = []
    for c in chunks[:12]:
        title = c.get('title', 'Unknown')
        year = c.get('year', '')
        genre = c.get('genre', '')
        text = c.get('text', '')
        ctx_parts.append(f"Title: {title} ({year})\nGenre: {genre}\n{text}")

    context_text = '\n\n'.join(ctx_parts)

    system_content = (
        "You are an expert assistant answering questions about a provided movie dataset. "
        "Only use the information in the provided CONTEXT. Do not invent facts. If the CONTEXT does not contain the answer, reply: 'No information in the provided context.' "
        "Be concise. When listing movies, always include title and genre and (if available) year. Prepend a one-line summary with the count (e.g., '5 movies:'). "
        "Format movie items as: '1. Title — Genre (Year) [source: Title]'. For distinct lists (genres/directors/actors/years) return a bulleted list."
    )

    user_content = (
        f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n"
        "1) Answer only from CONTEXT.\n"
        "2) If the question requests specific fields (e.g., 'movie name and genre'), output only those fields per item.\n"
        "3) Start with a summary line showing the count (e.g., '3 movies:').\n"
        "4) Format items as: '1. Title — Genre (Year) [source: Title]' or for distinct lists: '- Genre'.\n"
        "5) If no match, reply: 'No information in the provided context.'\n"
        "6) Keep the answer under 200 words."
    )

    prompt = (
        f"System: {system_content}\n\n"
        f"User: {user_content}\n\n"
        "Assistant:"
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode('utf-8'))
            return body.get('response', '').strip() or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, ConnectionRefusedError, socket.timeout, OSError):
        return None


def answer_query(query, k=5):
    model, chunks, index = load_artifacts()

    # handle distinct-field queries directly
    distinct_field = _extract_distinct_field(query)
    if distinct_field:
        filters = _extract_query_filters(query)
        values = []
        seen = set()
        for chunk in chunks:
            if filters and not all(_matches_field(field, value, chunk) for field, value in filters.items()):
                continue
            value = chunk.get(distinct_field, '')
            if not value:
                continue
            normalized = _normalize_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(str(value))
        if values:
            values = sorted(values, key=lambda item: item.lower())
            return '\n'.join(values)

    # perform embedding search
    qvec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    search_k = len(chunks)
    D, I = index.search(qvec, k=search_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= int(idx) < len(chunks):
            chunk = chunks[int(idx)]
            boosted_score = float(dist) + _boost_score(chunk, query)
            results.append((boosted_score, chunk))
    results.sort(key=lambda item: item[0], reverse=True)

    filters = _extract_query_filters(query)
    if filters:
        ranked_chunks = [chunk for _, chunk in _rank_filtered_chunks(chunks, results, filters)]
    else:
        ranked_chunks = [chunk for _, chunk in results]

    start_time = time.perf_counter()
    text = _synthesize_with_ollama(query, ranked_chunks)
    latency = time.perf_counter() - start_time

    if text:
        return f"{text}\n\nLLM latency: {latency:.3f} seconds"

    return "Ollama did not return a response. Check that the Ollama server is running and the requested model is available."
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
        out_fields = _extract_requested_fields(q)
        for score, chunk in results:
            if out_fields:
                print(_format_chunk_fields(chunk, out_fields))
            else:
                print('Score:', score)
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
                out_fields = _extract_requested_fields(q)
                for score, chunk in results:
                    if out_fields:
                        print(_format_chunk_fields(chunk, out_fields))
                    else:
                        print('Score:', score)
                        print(format_chunk(chunk))
                        print('---')
        except EOFError:
            print('\nExiting.')


if __name__ == '__main__':
    main()