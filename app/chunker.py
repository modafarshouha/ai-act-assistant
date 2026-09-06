"""Splits provisions into chunks that fit the 512-token embedding window."""

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from app.config import (
    CHUNK_MIN_TOKENS,
    CHUNK_TARGET_TOKENS,
    EMBEDDING_MAX_TOKENS,
    Settings,
    get_settings,
)
from app.corpus import Provision, load_corpus

# [CLS] and [SEP].
SPECIAL_TOKEN_OVERHEAD = 2

# A line opening a numbered paragraph: "1.", "6a.", "12.". Trailing space is
# optional because the enumerator sometimes sits alone on its line, and
# requiring it dropped exactly the paragraphs the Omnibus inserted.
_PARAGRAPH_OPEN = re.compile(r"^(\d+[a-z]?)\.(?:\s+|$)")
_ENUMERATOR = re.compile(r"\d+[a-z]?")

# Sentence end followed by something that could start a new one. Semicolons and
# colons count too; legal drafting ends enumerated points with them, so they are
# the natural seams in a long article.
_SENTENCE_CANDIDATE = re.compile(r"[.;:]\s+(?=[\"'(\u2018\u201cA-Z0-9])")
_PRECEDING_WORD = re.compile(r"([\w.]+)$")

# Splitting after any of these severs a citation: "Art. 6(1)" and "No. 2016/679"
# are single references.
_ABBREVIATIONS = frozenset({"art", "arts", "no", "nos", "para", "paras", "e.g", "i.e"})


@lru_cache(maxsize=2)
def _load_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # With truncation on, every long chunk reports exactly 512 tokens. That is
    # the one number here that has to be honest.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def get_tokenizer(tokenizer_dir: Path | None = None) -> Tokenizer:
    tokenizer_dir = tokenizer_dir or get_settings().tokenizer_dir
    return _load_tokenizer(str(tokenizer_dir / "tokenizer.json"))


def count_tokens(text: str, tokenizer: Tokenizer | None = None) -> int:
    tokenizer = tokenizer or get_tokenizer()
    return len(tokenizer.encode(text).ids)


def _content_tokens(text: str, tokenizer: Tokenizer) -> int:
    """Token cost minus the per-sequence special tokens.

    Packing sums these instead of re-encoding the growing chunk every time,
    which keeps the corpus pass linear. Summing over-estimates a little, which
    is the safe direction: chunks land under budget, never over. Final sizes
    get measured for real in chunk_provision.
    """
    return count_tokens(text, tokenizer) - SPECIAL_TOKEN_OVERHEAD


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_key: str
    celex: str
    regulation: str
    kind: str
    number: str
    title: str
    paragraphs: tuple[str, ...]
    citation: str
    header: str
    body: str
    text: str
    token_count: int
    amended: bool
    is_current: bool
    content_hash: str

    def metadata(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "source_key": self.source_key,
            "celex": self.celex,
            "regulation": self.regulation,
            "kind": self.kind,
            "number": self.number,
            "title": self.title,
            "paragraphs": list(self.paragraphs),
            "citation": self.citation,
            "token_count": self.token_count,
            "amended": self.amended,
            "is_current": self.is_current,
            "content_hash": self.content_hash,
        }


@dataclass
class _Unit:
    """A numbered paragraph and the lines hanging off it."""

    paragraph: str | None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _is_real_boundary(text: str, position: int) -> bool:
    match = _PRECEDING_WORD.search(text[:position])
    if match is None:
        return False
    word = match.group(1).rstrip(".").lower()
    if word in _ABBREVIATIONS or _ENUMERATOR.fullmatch(word):
        return False
    # A single initial, as in "R. METSOLA".
    return not (len(word) == 1 and word.isalpha())


def split_sentences(text: str) -> list[str]:
    """Split into sentences without breaking legal citations.

    A naive split on full stops cuts "Art. 6(1)" in half and splits the digit
    groups in "EUR 35 000 000", leaving half a fine amount in the next chunk.
    """
    sentences: list[str] = []
    start = 0
    for candidate in _SENTENCE_CANDIDATE.finditer(text):
        cut = candidate.start()
        if not _is_real_boundary(text, cut):
            continue
        piece = text[start : cut + 1].strip()
        if piece:
            sentences.append(piece)
        start = candidate.end()
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _split_units(provision: Provision) -> list[_Unit]:
    units: list[_Unit] = []
    current = _Unit(paragraph=None)
    for line in provision.text.split("\n"):
        if not line.strip():
            continue
        match = _PARAGRAPH_OPEN.match(line)
        if match and provision.kind != "recital":
            if current.lines:
                units.append(current)
            current = _Unit(paragraph=match.group(1))
        current.lines.append(line)
    if current.lines:
        units.append(current)
    return units


def _hard_split(text: str, budget: int, tokenizer: Tokenizer) -> list[str]:
    """Last resort for a sentence longer than the window. Packs whole words."""
    pieces: list[str] = []
    current: list[str] = []
    running = 0
    for word in text.split():
        cost = _content_tokens(word, tokenizer)
        if current and running + cost > budget:
            pieces.append(" ".join(current))
            current, running = [word], cost
        else:
            current.append(word)
            running += cost
    if current:
        pieces.append(" ".join(current))
    return pieces


def _fragments(text: str, budget: int, tokenizer: Tokenizer) -> list[tuple[str, int]]:
    """Reduce text to pieces that each fit the budget, with their token costs."""
    total = _content_tokens(text, tokenizer)
    if total <= budget:
        return [(text, total)]

    pieces: list[tuple[str, int]] = []
    current = ""
    running = 0
    for sentence in split_sentences(text):
        cost = _content_tokens(sentence, tokenizer)
        if cost > budget:
            if current:
                pieces.append((current, running))
                current, running = "", 0
            pieces.extend(
                (piece, _content_tokens(piece, tokenizer))
                for piece in _hard_split(sentence, budget, tokenizer)
            )
            continue
        if current and running + cost > budget:
            pieces.append((current, running))
            current, running = sentence, cost
        else:
            current = f"{current} {sentence}".strip() if current else sentence
            running += cost
    if current:
        pieces.append((current, running))
    return pieces


def _build_header(provision: Provision) -> str:
    parts = [provision.citation]
    if provision.title:
        parts.append(provision.title)
    header = " — ".join(parts)
    if not provision.is_current_text:
        header += " [original text, superseded]"
    elif provision.amended:
        header += " [amended]"
    return header


def _cite(provision: Provision, paragraphs: tuple[str, ...]) -> str:
    base = provision.citation
    if not paragraphs or provision.kind != "article":
        return base
    if len(paragraphs) == 1:
        return f"{base}({paragraphs[0]})"
    return f"{base}({paragraphs[0]})-({paragraphs[-1]})"


def chunk_provision(
    provision: Provision,
    settings: Settings | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    settings = settings or get_settings()
    tokenizer = tokenizer or get_tokenizer(settings.tokenizer_dir)

    header = _build_header(provision)
    header_tokens = count_tokens(f"{header}\n\n", tokenizer)
    budget = CHUNK_TARGET_TOKENS - header_tokens
    hard_budget = EMBEDDING_MAX_TOKENS - header_tokens
    if budget <= 0:
        return []

    packed: list[tuple[list[str], list[str]]] = []
    current: list[str] = []
    paragraphs: list[str] = []
    running = 0

    for unit in _split_units(provision):
        for fragment, cost in _fragments(unit.text, budget, tokenizer):
            if current and running + cost > budget:
                packed.append((current, paragraphs))
                current, paragraphs, running = [], [], 0
            current.append(fragment)
            running += cost
            if unit.paragraph and unit.paragraph not in paragraphs:
                paragraphs.append(unit.paragraph)
    if current:
        packed.append((current, paragraphs))

    # A trailing scrap of a few tokens retrieves badly alone. Fold it back if
    # the previous chunk has room.
    if len(packed) > 1:
        tail_lines, tail_paragraphs = packed[-1]
        tail = "\n".join(tail_lines).strip()
        previous = "\n".join(packed[-2][0]).strip()
        merged = f"{previous}\n{tail}"
        if _content_tokens(tail, tokenizer) < CHUNK_MIN_TOKENS and (
            _content_tokens(merged, tokenizer) <= hard_budget
        ):
            packed[-2] = (packed[-2][0] + tail_lines, packed[-2][1] + tail_paragraphs)
            packed.pop()

    chunks: list[Chunk] = []
    prefix = {"article": "art", "annex": "anx", "recital": "rct"}[provision.kind]
    for lines, found in packed:
        body = "\n".join(lines).strip()
        if not body:
            continue
        # Packing used estimates, so measure properly here. Anything still over
        # the window gets re-split; the embedder would truncate it in silence.
        for piece in _fragments(body, hard_budget, tokenizer):
            ordinal = len(chunks)
            text = f"{header}\n\n{piece[0]}"
            chunks.append(
                Chunk(
                    chunk_id=f"{provision.source_key}:{prefix}_{provision.number}:{ordinal}",
                    source_key=provision.source_key,
                    celex=provision.celex,
                    regulation=provision.regulation,
                    kind=provision.kind,
                    number=provision.number,
                    title=provision.title,
                    paragraphs=tuple(found),
                    citation=_cite(provision, tuple(found)),
                    header=header,
                    body=piece[0],
                    text=text,
                    token_count=count_tokens(text, tokenizer),
                    amended=provision.amended,
                    is_current=provision.is_current_text,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
    return chunks


def build_chunks(settings: Settings | None = None) -> list[Chunk]:
    settings = settings or get_settings()
    tokenizer = get_tokenizer(settings.tokenizer_dir)
    chunks: list[Chunk] = []
    for provision in load_corpus(settings):
        chunks.extend(chunk_provision(provision, settings, tokenizer))
    return chunks
