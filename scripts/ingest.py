"""Builds the hybrid index from the vendored corpus.

    python scripts/ingest.py            # skips if the corpus has not changed
    python scripts/ingest.py --force
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunker import build_chunks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embed import embed_documents  # noqa: E402
from app.index import HybridIndex  # noqa: E402


def fingerprint(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(chunk.content_hash.encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    args = parser.parse_args()

    settings = get_settings()
    manifest_path = settings.data_dir / "manifest.json"
    started = time.perf_counter()

    print("chunking the corpus...")
    chunks = build_chunks(settings)
    current = fingerprint(chunks)
    print(f"  {len(chunks)} chunks")

    if not args.force and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("corpus_fingerprint") == current:
            try:
                HybridIndex.load(settings.index_dir, settings)
            except (FileNotFoundError, OSError):
                pass
            else:
                print("index is current; nothing to do (use --force to rebuild)")
                return 0

    total = len(chunks)

    def progress(done: int, _: int) -> None:
        print(f"\r  embedding {done}/{total}", end="", flush=True)

    print("embedding (this takes a few minutes on CPU)...")
    vectors = embed_documents([chunk.text for chunk in chunks], settings, on_progress=progress)
    print()

    index = HybridIndex.build(chunks, vectors, settings)
    index.save(settings.index_dir)

    counts = [chunk.token_count for chunk in chunks]
    manifest_path.write_text(
        json.dumps(
            {
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "corpus_fingerprint": current,
                "chunks": len(chunks),
                "max_tokens": max(counts),
                "median_tokens": sorted(counts)[len(counts) // 2],
                "embedding_model": settings.embedding_model,
                "python": platform.python_version(),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {settings.index_dir} in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
