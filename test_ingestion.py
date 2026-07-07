import pytest

from ingestion import embed_texts, parse_document, read_text
from retrieval import format_chunk


def test_parse_document_splits_plain_text_into_chunks():
    text = 'First paragraph about a document.\n\nSecond paragraph with more context.'
    records = parse_document(text)

    assert len(records) == 2
    assert records[0]['title'] == 'First paragraph about a document.'
    assert 'Second paragraph' in records[1]['text']


def test_format_chunk_does_not_repeat_title_when_text_matches():
    assert format_chunk({'title': 'Summary', 'text': 'Summary'}) == 'Summary'


def test_embed_texts_uses_configured_timeout(monkeypatch):
    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"embeddings": [[0.1, 0.2]]}'

    captured = {}

    def fake_urlopen(request, timeout):
        captured['timeout'] = timeout
        return DummyResponse()

    monkeypatch.setenv('OLLAMA_TIMEOUT', '600')
    monkeypatch.setattr('ingestion.urllib.request.urlopen', fake_urlopen)

    embeddings = embed_texts(['hello'])

    assert embeddings.shape == (1, 2)
    assert captured['timeout'] == 600


@pytest.mark.skipif(False, reason='PDF parser dependency is optional at runtime')
def test_read_text_supports_pdf(tmp_path):
    pdf_bytes = b'%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R>>endobj\n2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1>>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n4 0 obj<< /Length 44 >>stream\nBT /F1 24 Tf 72 72 Td (Hello PDF) Tj ET\nendstream\nendobj\n5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n0000000119 00000 n \n0000000207 00000 n \n0000000304 00000 n \ntrailer<< /Size 6 /Root 1 0 R>>\nstartxref\n0\n%%EOF'
    pdf_path = tmp_path / 'sample.pdf'
    pdf_path.write_bytes(pdf_bytes)

    text = read_text(str(pdf_path))
    assert 'Hello PDF' in text
