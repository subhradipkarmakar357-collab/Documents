import argparse
import pickle
import re

import faiss
import os
import json
try:
    import openai
except Exception:
    openai = None
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


def _extract_year_range(query):
    # patterns like 'between 1946 and 1950', 'from 1946 to 1950'
    match = re.search(r'\bbetween\s+(19\d{2}|20\d{2})\s+and\s+(19\d{2}|20\d{2})\b', query, re.IGNORECASE)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return (min(start, end), max(start, end))

    match = re.search(r'\bfrom\s+(19\d{2}|20\d{2})\s+(?:to|until|through)\s+(19\d{2}|20\d{2})\b', query, re.IGNORECASE)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
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
    if not re.search(r'\b(?:distinct|unique|all|list|show|give|names? of|which are|what are)\b', query_lower):
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
    years = re.findall(r'\b(?:19|20)\d{2}\b', text_fields)
    if not years:
        return False

    start, end = year_range
    for y in years:
        try:
            yv = int(y)
        except Exception:
            continue
        if start <= yv <= end:
            return True
    return False


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
def retrieve_file(query, k=5):
    model, chunks, index = load_artifacts()

    distinct_field = _extract_distinct_field(query)
    if distinct_field:
        values = []
        seen = set()
        for chunk in chunks:
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
    search_k = min(len(chunks), max(k * 3, 20))
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
        strict_matches = [
            (score, chunk)
            for score, chunk in results
            if all(_matches_field(field, value, chunk) for field, value in filters.items())
        ]
        if strict_matches:
            return strict_matches

        strict_matches_all = [
            (0.0, chunk)
            for chunk in chunks
            if all(_matches_field(field, value, chunk) for field, value in filters.items())
        ]
        if strict_matches_all:
            return strict_matches_all

    return results


def _synthesize_with_openai(question, chunks):
    if not openai or not os.getenv('OPENAI_API_KEY'):
        return None

    # Build a focused CONTEXT from the top chunks
    ctx_parts = []
    for c in chunks[:12]:
        title = c.get('title', 'Unknown')
        year = c.get('year', '')
        genre = c.get('genre', '')
        text = c.get('text', '')
        # concise context lines per record
        ctx_parts.append(f"Title: {title} ({year})\nGenre: {genre}\n{ text }")

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

    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content},
    ]

    try:
        model_name = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        resp = openai.ChatCompletion.create(model=model_name, messages=messages, max_tokens=400, temperature=0.0)
        return resp['choices'][0]['message']['content'].strip()
    except Exception:
        return None


def _synthesize_fallback(question, chunks):
    # Deterministic fallback: if query includes year_range or year filters, prefer those
    filters = _extract_query_filters(question)
    if 'year_range' in filters or 'year' in filters:
        # use the strict matches behavior from retrieve_file
        matches = []
        for chunk in chunks:
            if all(_matches_field(f, v, chunk) for f, v in filters.items()):
                matches.append(chunk)
        if matches:
            lines = [f"{m.get('title')} — {m.get('genre', 'Unknown')}" for m in matches]
            return f"{len(matches)} movies:\n" + '\n'.join(lines)

    # Generic fallback: summarize top-k chunks
    top = chunks[:5]
    if not top:
        return 'No relevant information found.'
    lines = []
    for c in top:
        title = c.get('title') or c.get('text')[:60]
        genre = c.get('genre', '')
        lines.append(f"- {title}{f' — {genre}' if genre else ''}")
    return 'Answer based on top documents:\n' + '\n'.join(lines)


def answer_query(query, k=5):
    model, chunks, index = load_artifacts()

    # handle distinct-field queries directly
    distinct_field = _extract_distinct_field(query)
    if distinct_field:
        values = []
        seen = set()
        for chunk in chunks:
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
    search_k = min(len(chunks), max(k * 3, 20))
    D, I = index.search(qvec, k=search_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= int(idx) < len(chunks):
            chunk = chunks[int(idx)]
            boosted_score = float(dist) + _boost_score(chunk, query)
            results.append((boosted_score, chunk))
    results.sort(key=lambda item: item[0], reverse=True)
    ranked_chunks = [c for _, c in results]

    # Try OpenAI first
    text = _synthesize_with_openai(query, ranked_chunks)
    if text:
        return text

    # Fallback deterministic synthesizer
    return _synthesize_fallback(query, ranked_chunks)
def main():
    parser = argparse.ArgumentParser(description='Query the embedding-based FAISS index of documents')
    parser.add_argument('query', nargs='*', help='Query text (if omitted, enters interactive prompt)')
    parser.add_argument('-k', '--k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--answer', action='store_true', help='Return synthesized answer using an LLM instead of raw chunks')
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