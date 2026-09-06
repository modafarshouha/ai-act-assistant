import pytest

from app.chunker import build_chunks, chunk_provision, count_tokens, split_sentences
from app.config import EMBEDDING_MAX_TOKENS


@pytest.fixture(scope="module")
def chunks(settings):
    return build_chunks(settings)


def test_no_chunk_exceeds_the_embedding_window(chunks):
    over = [c.chunk_id for c in chunks if c.token_count > EMBEDDING_MAX_TOKENS]
    assert not over, f"{len(over)} chunks would be silently truncated: {over[:5]}"


def test_token_counts_are_measured_not_estimated(chunks):
    sample = chunks[len(chunks) // 2]
    assert sample.token_count == count_tokens(sample.text)


def test_chunks_never_span_two_provisions(chunks):
    for chunk in chunks:
        assert chunk.chunk_id.startswith(chunk.source_key)
        assert chunk.citation.startswith(chunk.regulation)


def test_every_chunk_carries_its_citation_in_the_text(chunks):
    for chunk in chunks[:200]:
        assert chunk.header in chunk.text
        assert chunk.regulation in chunk.header


def test_superseded_provisions_are_marked_in_the_header(chunks):
    originals = [c for c in chunks if c.source_key == "ai_act_original"]
    assert originals
    assert all("superseded" in c.header for c in originals)


def test_article_99_splits_on_paragraph_boundaries(provisions, settings):
    article = next(
        p for p in provisions if p.source_key == "ai_act_consolidated" and p.number == "99"
    )
    pieces = chunk_provision(article, settings)
    assert len(pieces) > 1
    assert all(piece.paragraphs for piece in pieces)
    # The SMC modifier lives in 99(6a) and must stay findable by citation.
    assert any("6a" in piece.paragraphs for piece in pieces)


def test_sentence_split_keeps_article_references_whole():
    text = "See Art. 6(1) for the rule. The next sentence follows."
    assert len(split_sentences(text)) == 2


def test_sentence_split_keeps_grouped_amounts_whole():
    sentences = split_sentences("Fines up to EUR 35 000 000 apply. That is the ceiling.")
    assert "EUR 35 000 000" in sentences[0]


def test_paragraph_enumerators_do_not_end_a_sentence():
    assert len(split_sentences("6a. The cap applies to SMCs.")) == 1
