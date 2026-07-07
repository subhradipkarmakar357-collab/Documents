from pathlib import Path

from ingestion import parse_document
from retrieval import (
    _extract_director,
    _extract_actors,
    _extract_genre,
    _extract_release_date,
    _extract_year,
    _extract_distinct_field,
    _matches_director,
    _matches_actors,
    _matches_genre,
    _matches_release_date,
    _matches_year,
    format_chunk,
)


def test_parse_document_creates_movie_records():
    text = Path('OscarMovies').read_text(encoding='utf-8')
    records = parse_document(text)

    assert len(records) >= 3
    assert any(record['title'] == 'Gone with the Wind' for record in records)
    assert any('Director:' in record['text'] for record in records)


def test_director_query_matches_director_metadata():
    chunk = {
        'title': 'Rebecca',
        'director': 'Alfred Hitchcock',
    }

    assert _extract_director('which movie was directed by Alfred Hitchcock') == 'Alfred Hitchcock'
    assert _matches_director('Alfred Hitchcock', chunk) is True


def test_director_query_does_not_match_same_first_name():
    chunk = {
        'title': 'Cavalcade',
        'director': 'Frank Lloyd',
    }

    assert _extract_director('which films are directed by Frank Capra') == 'Frank Capra'
    assert _matches_director('Frank Capra', chunk) is False


def test_actor_query_matches_actor_metadata():
    chunk = {
        'title': 'Gone with the Wind',
        'actors': 'Vivien Leigh, Clark Gable',
    }

    assert _extract_actors('which movie starred Clark Gable') == 'Clark Gable'
    assert _matches_actors('Clark Gable', chunk) is True


def test_genre_query_matches_genre_metadata():
    chunk = {
        'title': 'It Happened One Night',
        'genre': 'Romantic Comedy',
    }

    assert _extract_genre('which movie is a romantic comedy') == 'romantic comedy'
    assert _matches_genre('romantic comedy', chunk) is True


def test_release_date_and_year_query_matches_metadata():
    chunk = {
        'title': 'Casablanca',
        'release_date': 'November 26, 1943',
        'year': '1943-44',
    }

    assert _extract_release_date('which film was released in 1943') == '1943'
    assert _matches_year('1943', chunk) is True
    assert _matches_release_date('in 1943', chunk) is True


def test_distinct_genre_query_is_detected():
    assert _extract_distinct_field('give me a list of all distinct genres in the oscar list') == 'genre'


def test_distinct_director_and_actor_queries_are_detected():
    assert _extract_distinct_field('list all distinct directors') == 'director'
    assert _extract_distinct_field('show unique actors') == 'actors'


def test_format_chunk_does_not_repeat_title_when_text_matches():
    assert format_chunk({'title': 'Drama', 'text': 'Drama'}) == 'Drama'
