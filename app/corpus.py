"""Parses the vendored EUR-Lex XHTML into articles, annexes and recitals."""

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.config import Settings, get_settings

UnitKind = Literal["article", "annex", "recital"]


@dataclass(frozen=True)
class Source:
    key: str
    celex: str
    regulation: str
    is_current: bool


SOURCES = (
    Source("ai_act_consolidated", "02024R1689-20260727", "AI Act", True),
    Source("gdpr", "32016R0679", "GDPR", True),
    # Indexed so "what did it say before the Omnibus?" works. The currency
    # filter in index.fuse keeps it out of ordinary answers.
    Source("ai_act_original", "32024R1689", "AI Act (original)", False),
    Source("omnibus", "32026R1744", "Digital Omnibus", True),
)

SOURCES_BY_KEY = {source.key: source for source in SOURCES}

# Ends on the closing quote, otherwise it also matches nested ids like
# "art_99.tit_1".
_PROVISION_OPEN = re.compile(
    r'<div\b[^>]*\bid="(?P<id>art_(?P<art>[0-9]+[a-z]*)|anx_(?P<anx>[IVXLC]+)|rct_(?P<rct>[0-9]+))"'
)
_DIV_TAG = re.compile(r"<(/?)div\b")

# Consolidated texts mark amended passages with ▼M1 / ►M1 ... ◄.
_AMENDMENT_MARKER = re.compile(r"[\u25bc\u25ba]\s*([BM]\d*)")
_MARKER_CHARS = re.compile(r"[\u25bc\u25ba\u25c4]\s*[BM]?\d*")

_BLOCK_END = re.compile(r"</(p|div|td|tr|th|li|h[1-6])\s*>|<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_ARTICLE_HEADING = re.compile(r"^Article\s+[0-9]+[a-z]*$", re.IGNORECASE)
_ANNEX_HEADING = re.compile(r"^ANNEX\s+[IVXLC]+$", re.IGNORECASE)

# "(a)", "(iii)", "3.", "6a.". The source puts these in their own table cell,
# so they arrive alone on a line.
_LONE_ENUMERATOR = re.compile(r"^(\(\w{1,5}\)|\d+[a-z]?\.)$")

_SPACES = str.maketrans({"\u00a0": " ", "\u2007": " ", "\u202f": " "})


@dataclass(frozen=True)
class Provision:
    source_key: str
    regulation: str
    kind: UnitKind
    number: str
    title: str
    text: str
    amended: bool

    @property
    def citation(self) -> str:
        if self.kind == "article":
            return f"{self.regulation} Art. {self.number}"
        if self.kind == "annex":
            return f"{self.regulation} Annex {self.number}"
        return f"{self.regulation} Recital {self.number}"

    @property
    def anchor(self) -> str:
        prefix = {"article": "art", "annex": "anx", "recital": "rct"}[self.kind]
        return f"{self.source_key}:{prefix}_{self.number}"

    @property
    def celex(self) -> str:
        return SOURCES_BY_KEY[self.source_key].celex

    @property
    def is_current_text(self) -> bool:
        return SOURCES_BY_KEY[self.source_key].is_current


def normalize(text: str) -> str:
    """Collapse whitespace so quoted provisions can be found by substring search.

    Legal XHTML is full of non-breaking spaces and hard wrapping, so a search
    for "Article 99" fails on whitespace alone without this.
    """
    return re.sub(r"\s+", " ", text.translate(_SPACES)).strip()


def _strip_markup(fragment: str) -> str:
    text = _BLOCK_END.sub("\n", fragment)
    text = _TAG.sub(" ", text)
    text = html.unescape(text).translate(_SPACES)
    text = _MARKER_CHARS.sub(" ", text)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    # Rejoin an enumerator with its clause. Otherwise a chunk opens on a bare
    # "(a)" and carries on mid-sentence.
    joined: list[str] = []
    for line in lines:
        if joined and _LONE_ENUMERATOR.match(joined[-1]):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return "\n".join(joined)


def _balanced_extent(raw: str, start: int, after_open: int) -> str:
    depth = 1
    end = len(raw)
    for tag in _DIV_TAG.finditer(raw, after_open):
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            end = tag.start()
            break
    return raw[start:end]


def _heading_and_title(lines: list[str], kind: UnitKind) -> tuple[str, list[str]]:
    """Strip the heading off the body. number and title already hold it.

    The chunker prepends it as a header anyway, so leaving it here repeats
    "Article 99 / Penalties" inside every chunk of Article 99.
    """
    if kind == "recital" or not lines:
        return "", lines
    matcher = _ARTICLE_HEADING if kind == "article" else _ANNEX_HEADING
    if not matcher.match(lines[0]):
        return "", lines
    if len(lines) > 1 and not _LONE_ENUMERATOR.match(lines[1]):
        return lines[1], lines[2:]
    return "", lines[1:]


def parse_provisions(raw: str, source: Source) -> list[Provision]:
    provisions: list[Provision] = []
    seen: set[str] = set()

    for match in _PROVISION_OPEN.finditer(raw):
        identifier = match.group("id")
        if identifier in seen:
            continue
        seen.add(identifier)

        if match.group("art") is not None:
            kind: UnitKind = "article"
            number = match.group("art")
        elif match.group("anx") is not None:
            kind, number = "annex", match.group("anx")
        else:
            kind, number = "recital", match.group("rct")

        fragment = _balanced_extent(raw, match.start(), match.end())
        body = _strip_markup(fragment)
        if not body:
            continue

        lines = body.split("\n")
        title, lines = _heading_and_title(lines, kind)
        provisions.append(
            Provision(
                source_key=source.key,
                regulation=source.regulation,
                kind=kind,
                number=number,
                title=title,
                text="\n".join(lines),
                amended=bool(_AMENDMENT_MARKER.findall(fragment)),
            )
        )

    return provisions


def load_raw(source_key: str, raw_dir: Path | None = None) -> str:
    raw_dir = raw_dir if raw_dir is not None else get_settings().raw_dir
    return (raw_dir / f"{source_key}.xhtml").read_text(encoding="utf-8")


def load_corpus(settings: Settings | None = None) -> list[Provision]:
    settings = settings or get_settings()
    provisions: list[Provision] = []
    for source in SOURCES:
        provisions.extend(parse_provisions(load_raw(source.key, settings.raw_dir), source))
    return provisions
