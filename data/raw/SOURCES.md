# Corpus sources

All four documents are committed in this directory. Nothing is fetched at
runtime, so the container build needs no network and the evaluation gives the
same answer every time.

## How these were obtained

EUR-Lex's HTML endpoint sits behind a bot challenge that answers HTTP 202 with a
zero-byte body, so it is unusable. These came from the Publications Office
Cellar, addressed by manifestation UUID:

    GET https://publications.europa.eu/resource/cellar/{uuid}.{expr}.{manif}/DOC_1

Manifestation `.03` is the XHTML one. XHTML over PDF because the
`eli-subdivision` and `oj-ti-art` markup delimits articles structurally, so a
citation can resolve to "AI Act Art. 99(3)" instead of a page number.

The URLs below are the source of record. Sizes and SHA-256 values are recorded
per document for provenance; check one by hand with
`sha256sum data/raw/gdpr.xhtml` if you want to confirm nothing has drifted.

## Licence and reuse

© European Union, 1998-2026. Source: [EUR-Lex](https://eur-lex.europa.eu),
reused under Commission Decision 2011/833/EU on the reuse of Commission
documents. Consolidated texts and editorial content are additionally licensed
under CC BY 4.0; EUR-Lex metadata is dedicated to the public domain under
CC0 1.0. Logos and trademarks are excluded from this permission.

> Only the legislation published in the printed edition of the Official Journal
> of the European Union is deemed authentic.

The disclaimer is EUR-Lex's own. It is repeated here because this repo computes
monetary figures someone might act on.

## Documents


### Regulation (EU) 2024/1689 (AI Act), consolidated as of 27/07/2026

- **Key:** `ai_act_consolidated` (vendored as `ai_act_consolidated.xhtml`)
- **CELEX:** `02024R1689-20260727`
- **Cellar manifestation:** `b1730fb2-8f1c-11f1-9262-01aa75ed71a1.0001.03`
- **URL:** https://publications.europa.eu/resource/cellar/b1730fb2-8f1c-11f1-9262-01aa75ed71a1.0001.03/DOC_1
- **Retrieved:** 2026-09-04
- **Size:** 851,286 bytes
- **SHA-256:** `5e7719f77e8a606b257dc25958ee3222c4383300a5a34270a5b850a2ce8b8715`
- **Role in the corpus:** Current law. Incorporates the Digital Omnibus
  amendments, including the deferred high-risk dates and the Art. 99(6a) cap.

### Regulation (EU) 2016/679 (GDPR)

- **Key:** `gdpr` (vendored as `gdpr.xhtml`)
- **CELEX:** `32016R0679`
- **Cellar manifestation:** `3e485e15-11bd-11e6-ba9a-01aa75ed71a1.0006.03`
- **URL:** https://publications.europa.eu/resource/cellar/3e485e15-11bd-11e6-ba9a-01aa75ed71a1.0006.03/DOC_1
- **Retrieved:** 2026-09-04
- **Size:** 806,864 bytes
- **SHA-256:** `962539af03738bf552319ff4ce42d69e5f95a576307c4dfed7bf87e81b646b9d`
- **Role in the corpus:** Current law. Article 83 administrative fines.

### Regulation (EU) 2024/1689 (AI Act), original as published 12/07/2024

- **Key:** `ai_act_original` (vendored as `ai_act_original.xhtml`)
- **CELEX:** `32024R1689`
- **Cellar manifestation:** `dc8116a1-3fe6-11ef-865a-01aa75ed71a1.0006.03`
- **URL:** https://publications.europa.eu/resource/cellar/dc8116a1-3fe6-11ef-865a-01aa75ed71a1.0006.03/DOC_1
- **Retrieved:** 2026-09-04
- **Size:** 1,262,391 bytes
- **SHA-256:** `8f0b656302f9864cc87e040c371f209a9d65ae1a6cecc25ca5eb737e872d721a`
- **Role in the corpus:** Point-in-time reference. Kept so the assistant can
  answer "what did the law say before the amendment", with the superseded dates
  quotable from source.

### Regulation (EU) 2026/1744 (Digital Omnibus on AI)

- **Key:** `omnibus` (vendored as `omnibus.xhtml`)
- **CELEX:** `32026R1744`
- **Cellar manifestation:** `b459c07f-86fb-11f1-bf5e-01aa75ed71a1.0006.03`
- **URL:** https://publications.europa.eu/resource/cellar/b459c07f-86fb-11f1-bf5e-01aa75ed71a1.0006.03/DOC_1
- **Retrieved:** 2026-09-04
- **Size:** 351,287 bytes
- **SHA-256:** `9d754652b867722807e4219c85912ce354233e58a1b4eb8c7752b4d1922993db`
- **Role in the corpus:** The amending act. In force 27 July 2026, and the
  reason a model working from memory gives the old high-risk dates.
