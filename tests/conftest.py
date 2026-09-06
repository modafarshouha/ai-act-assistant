import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import build_graph  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.corpus import SOURCES, load_corpus, load_raw, normalize  # noqa: E402
from app.index import HybridIndex  # noqa: E402
from app.llm import DummyLLM  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def index(settings):
    try:
        return HybridIndex.load(settings.index_dir, settings)
    except FileNotFoundError:
        pytest.skip("no index; run scripts/ingest.py")


@pytest.fixture(scope="session")
def graph(index, settings):
    return build_graph(DummyLLM(), index, settings)


@pytest.fixture(scope="session")
def provisions(settings):
    return load_corpus(settings)


@pytest.fixture(scope="session")
def source_text(settings):
    """Whole documents, whitespace-normalised, for locating quoted facts."""
    return {
        source.key: normalize(load_raw(source.key, settings.raw_dir)) for source in SOURCES
    }
