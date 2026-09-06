from app.config import RRF_K
from app.index import tokenize


def test_bm25_tokenizer_keeps_legal_references_whole():
    tokens = tokenize("Article 6(2) and Regulation 2016/679")
    assert "6(2)" in tokens
    assert "2016/679" in tokens


def test_rrf_rewards_agreement_between_retrievers(index):
    dense = [(0, 0.9), (1, 0.8)]
    sparse = [(2, 5.0), (1, 4.0)]
    hits = index.fuse(dense, sparse, k=3, include_superseded=True)
    # Chunk 1 is second in both lists and first in neither, and still wins.
    # Agreement beating one strong vote is the point of fusing on rank.
    assert hits[0].chunk is index.chunks[1]
    assert hits[0].dense_rank == 2 and hits[0].sparse_rank == 2


def test_rrf_score_matches_the_formula(index):
    hits = index.fuse([(0, 0.9)], [], k=1, include_superseded=True)
    assert hits[0].score == 1.0 / (RRF_K + 1)


def test_the_currency_filter_drops_superseded_chunks(index):
    superseded = [i for i, c in enumerate(index.chunks) if not c.is_current]
    assert superseded, "the original AI Act should be indexed"
    ids = [(i, 1.0) for i in superseded[:5]]
    assert index.fuse(ids, [], k=5, include_superseded=False) == []
    assert len(index.fuse(ids, [], k=5, include_superseded=True)) == 5


def test_recitals_are_down_weighted_against_articles(index):
    recital = next(i for i, c in enumerate(index.chunks) if c.kind == "recital")
    article = next(i for i, c in enumerate(index.chunks) if c.kind == "article")
    hits = index.fuse([(recital, 1.0), (article, 0.9)], [], k=2, include_superseded=True)
    assert hits[0].chunk.kind == "article"


def test_dense_search_returns_scored_ids(index):
    from app.embed import embed_query

    hits = index.search_dense(embed_query("administrative fines"), 5)
    assert len(hits) == 5
    assert hits == sorted(hits, key=lambda hit: -hit[1])


def test_sparse_search_finds_an_exact_identifier(index):
    hits = index.search_sparse("Article 99 penalties", 10)
    citations = [index.chunks[i].citation for i, _ in hits]
    assert any("Art. 99" in citation for citation in citations)


def test_round_trip_preserves_chunks(index, tmp_path):
    from app.index import HybridIndex

    index.save(tmp_path)
    reloaded = HybridIndex.load(tmp_path, index.settings)
    assert len(reloaded) == len(index)
    assert reloaded.chunks[0].text == index.chunks[0].text
    assert reloaded.chunks[0].is_current == index.chunks[0].is_current
