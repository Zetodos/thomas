# Licensing & provenance audit — Gospel of Thomas reconstruction resources

**Scope:** Redistributability for an offline open-source repo (`Zetodos/thomas`).
**Method:** Public web / primary license pages / corpus READMEs (2026-11-09).
**Updates:** JAK-13–19 decisions recorded as **Done** (see § Decisions). Sahidic NT: Horner / CrossWire `CopSahHorner` (JAK-14; implementation JAK-24). Greek NT default: Westcott–Hort (JAK-15); SBLGNT revisit JAK-25.
**Caveat:** Not legal advice. Licenses change; pin versions and re-check before bundling. Confidence tags: **High** (primary license text seen), **Medium** (reputable secondary + consistent primary snippets), **Low** (inference / conflicting claims / no clear instrument).

Setup modes used below:

| Mode | Meaning |
| --- | --- |
| **bundled** | Safe to ship in git (with attribution/NOTICE) |
| **setup-download** | Fetch during explicit `thomas setup` from a known URL; not committed |
| **manual-import** | User supplies files; repo documents expected paths/checksums only |

Project policy derived from this audit: [`source-licenses.md`](source-licenses.md).

---

## 1. Coptic SCRIPTORIUM Dilley Gospel of Thomas

| Field | Finding |
| --- | --- |
| **License status** | Document/corpus metadata and SCRIPTORIUM practice: **CC BY 4.0** for `thomas.gospel` (Dilley edition). SCRIPTORIUM corpora README: default **CC BY 3.0 or 4.0** unless excepted; Thomas is not listed among exceptions. |
| **Redistributable** | **Yes** (under CC BY) |
| **Attribution** | Credit Coptic SCRIPTORIUM; Paul C. Dilley (edition/primary annotations); note contributors (Schroeder, Bremer-McCollum, Zeldes); version (e.g. v6.2.0); link license; mark modifications. Cite Dilley’s editorial note that restorations consult Layton 1989 and Bethge et al. 1996 **without redistributing those copyrighted editions**. |
| **Setup mode** | **bundled** (preferred for core Coptic) or **setup-download** from GitHub/Zenodo (`doi:10.5281/zenodo.17917497`) |
| **Blocking risks** | Low for Dilley open text. Do **not** substitute Layton (Brill) or Bethge (DBG *Synopsis*) full texts. Dilley uses Nag Hammadi facsimile images as sources — images themselves are not the license grant. |
| **Confidence** | **High** for CC BY on Thomas corpus metadata / SCRIPTORIUM README defaults |

**Primary refs:**
- https://github.com/CopticScriptorium/corpora (README “Sources and licenses”)
- Thomas ANNIS metadata: `license CC-BY 4.0`, `Coptic_edition Paul Dilley`, `project Coptic SCRIPTORIUM, Dilley`
- https://data.copticscriptorium.org/texts/thomasgospel/gospel-of-thomas/

---

## 2. P.Oxy. 1, 654, 655 (Greek Thomas fragments)

| Field | Finding |
| --- | --- |
| **License status** | **Grenfell & Hunt** editio princeps (1897 Logia; P.Oxy. I 1898; P.Oxy. IV 1904): **public domain** in US/EU for those print editions (authors deceased; pre-1929 US publications). **Hugh G. Evelyn-White**, *The Sayings of Jesus from Oxyrhynchus* (Cambridge, **1920**): **US public domain** for that original edition (pre-1929 US publication). **Later / modern critical editions** (Attridge in Layton NHS 20 / Brill; Fitzmyer article; Bethge apparatus; and any post-1928 copyrighted re-editions or modern editorial layers): **copyrighted — not redistributable**. **DCLP / papyri.info** TEI transcriptions: **CC BY 3.0** (`papyri/idp.data`). |
| **Redistributable** | **Yes** for PD Grenfell/Hunt text you rekey, Evelyn-White **1920** original, or DCLP TEI under CC BY 3.0. **No** for Attridge/Layton/Fitzmyer continuous critical text or other modern copyrighted editorial work. |
| **Attribution** | For DCLP: credit Digital Corpus of Literary Papyri / papyri.info + CC BY 3.0. Always record manuscript IDs (e.g. TM 62838–62840 / DCLP URIs). Separate **surviving letters** vs **editorial restorations**; cite which edition’s restorations if used. For Evelyn-White 1920: cite the 1920 Cambridge edition distinctly from any later apparatus. |
| **Setup mode** | **bundled** small excerpts from PD Grenfell/Hunt, Evelyn-White 1920, or DCLP (tiny corpus); or **setup-download** from `papyri/idp.data` |
| **Blocking risks** | Copying Attridge’s Greek appendix wholesale is a clear blocker. Do **not** conflate Evelyn-White **1920** (US PD) with later copyrighted critical editions. Website dumps that silently mix modern restorations without provenance. Prefer diplomatic + marked restorations. |
| **Confidence** | **High** for DCLP CC BY 3.0, for “do not redistribute Brill/Attridge”, and for US PD of Evelyn-White 1920. **Medium** for asserting PD of every Grenfell/Hunt digitization (encoding layer may add rights — prefer DCLP or self-OCR of PD scan). |

**Primary refs:**
- https://github.com/papyri/idp.data/blob/master/README.md (CC BY 3.0)
- Grenfell & Hunt 1897 / 1904 editions
- Evelyn-White 1920 (US PD original): https://archive.org/details/sayingsofjesusfr00eveluoft
- Do **not** treat Layton 1989 / Attridge appendix as open

---

## 3. Sahidica / Wells NT (gated — not default)

| Field | Finding |
| --- | --- |
| **License status** | **Custom Wells license**, not CC. SCRIPTORIUM explicitly excepts Sahidica NT from its default CC. Text (as published via SCRIPTORIUM / CrossWire / Wells): © J. Warren Wells (~2000–2008); “available free of charge for use in **free electronic editions of the New Testament** as long as the **full title and copyright information** are included and credited”; **print requires written permission**. Companion Greek **NTAEG** (“New Testament According to the Egyptian Greek Text”) is Wells’s parallel Greek (altered toward Sahidic with UBS apparatus support); **no separate clear open license found**; commercially sold via Faithlife/Logos historically — treat as **same risk family / uncertain**. |
| **Redistributable** | **Uncertain → treat as constrained.** A Thomas reconstruction repo that embeds Sahidica as **training/parallel data** may **not** fit the “free electronic editions of the New Testament” clause. Safer reading: **not OSI-open**; **not** freely re-licenseable. |
| **Project decision (JAK-14)** | **Do not bundle** Wells Sahidica, NTAEG, or Bohairica in default/git/published artifacts. Default Sahidic NT baseline is **Horner / CrossWire `CopSahHorner`** (see §4). Optional user-supplied import may be documented later without claiming permission. |
| **Attribution** | If ever used under verified rights: full title *Sahidica – A New Edition of the New Testament in Sahidic Coptic*; copyright notice; credit Wells; retain SCRIPTORIUM annotation provenance separately if annotations used. |
| **Setup mode** | **manual-import** only if rights confirmed; otherwise **out of default offline path**. Do not use `setup-download` as a workaround for redistribution limits. |
| **Blocking risks** | Ambiguous scope of Wells grant; print prohibition; NTAEG/Bohairica do not inherit Sahidica permission by inference; SCRIPTORIUM annotations do not erase underlying text rights. |
| **Confidence** | **High** for custom-license wording via SCRIPTORIUM’s dedicated license page. **Low–Medium** for whether Thomas-mapping use would be permitted. |

**Primary refs:**
- https://copticscriptorium.org/download/corpora/Mark/coptic_nt_sahidic.html
- CopticScriptorium/corpora README exceptions list
- CrossWire CopSahidica module about-text (same Wells terms)

---

## 4. Other Sahidic NT/OT parallel corpora (redistributable candidates)

| Source | License | Redistributable | Setup | Notes / confidence |
| --- | --- | --- | --- | --- |
| **CrossWire `CopSahHorner`** (G. W. Horner Sahidic NT, module v1.5) | CrossWire **DistributionLicense = Public Domain**; print edition 1911–1924 (Horner d. 1930) PD in US for pre-1929 / author-dead works | **Yes** for this CrossWire module (pin version/checksum) | **setup-download** (+ bundled PD pilot excerpt) | **Default Sahidic NT baseline (JAK-14).** Ingest/verify: JAK-24 — zip `sha256:4f074891…2132`; Unicode Coptic + overlines confirmed; John 7:53–8:11 / Mark 16:9–20 empty. See [`horner-wh-alignment-notes.md`](horner-wh-alignment-notes.md) and [`data/manifests/copsahhorner.json`](../data/manifests/copsahhorner.json). TextSource: Slavic Bible for Windows → CrossWire Unicode/OSIS. Quality caveats: eclectic MSS base; NRSV versification; historical verse misnumbering fixed in v1.5; weaker than Sahidica for Greek-parallel NLP. **High** confidence on CrossWire PD claim for the module. |
| **SCRIPTORIUM `sahidic.ot`** (Göttingen / CoptOT basetext via SCRIPTORIUM) | **CC BY-SA 4.0** (SCRIPTORIUM OT license page) | **Yes** (SA obligations) | **bundled** or **setup-download** | Strong OT parallel candidate. English alignment often WEB (PD). **High** confidence for CC BY-SA claim on SCRIPTORIUM page. |
| **SCRIPTORIUM Bohairic NT/OT / Wells Bohairica** | SCRIPTORIUM default CC BY 3.0/4.0; third-party notes also say CC BY-SA 4.0 for bohairic corpora. Underlying Wells **Bohairica** © 2006 “all rights reserved” with **unclear** electronic permission (unlike Sahidica’s explicit free-electronic clause). Marcion/Takla also in provenance chain. | **Uncertain** for full NT text | **`manual-import` only** after rights confirmation; **not** `setup-download` while unresolved; otherwise unavailable / out of default | Secondary dialect only per plan. Gated with Sahidica/NTAEG (JAK-14). **Medium** conflict between SCRIPTORIUM packaging and Wells ©. |
| **CoptOT manuscript transcriptions / critical edition** | Project DMP discusses publication; do not assume full critical edition dump is free to ship | **Uncertain** beyond SCRIPTORIUM OT release | Prefer SCRIPTORIUM `sahidic.ot` | **Medium** |
| **St Shenouda Society / PHI Brakke digital bases** | Proprietary / permissioned historically (Sahidica itself was based on them by permission) | **No** (assume) | **manual-import** only with rights | Do not scrape. **Medium** |
| **Greek side for OT parallels** | Pair Sahidic OT with **Swete LXX** (First1KGreek **CC BY-SA 4.0** digitization; 19th-c. text PD) or carefully handled Rahlfs 1935 (see §6) | Yes for Swete digitization under BY-SA | **bundled** / **setup-download** | ShareAlike may infect **adapted** parallel databases. **High** for First1KGreek license.md |

**Blocking risks (updated):** Lack of a Wells-free Sahidic NT is **resolved for licensing** by Horner/`CopSahHorner`. Remaining work is quality/coverage verification at ingest (JAK-24), not a Phase 0 license blocker.

**Primary refs (Horner):**
- https://www.crosswire.org/sword/modules/ModInfo.jsp?modName=CopSahHorner
- https://www.crosswire.org/sword/copyright/ModInfoCopyright.jsp?modName=CopSahHorner
- https://www.crosswire.org/ftpmirror/pub/sword/raw/mods.d/copsahhorner.conf

---

## 5. DIOSKOROS, Coptic lexicons, Greek loanword indexes

| Resource | License status | Redistributable | Setup | Risks / confidence |
| --- | --- | --- | --- | --- |
| **Comprehensive Coptic Lexicon (CCL) TEI** — BBAW Egyptian-Coptic + **DDGLC Greek loanwords** lexicon dump | **CC BY-SA 4.0** (coptic-dictionary.org/about; KELLIA/dictionary README; reFUbium DOI `10.17169/refubium-27566`) | **Yes** (ShareAlike on adaptations) | **bundled** or **setup-download** | **Default** loanword/lexicon path (JAK-19 Done). Attribute Burns/Feder/John/Kupreyev et al., TLA IDs. Do not wait on DIOSKOROS; no scrape. **High**. |
| **Coptic Dictionary Online interface** | Data CC BY-SA 4.0; interface software Apache-2.0 (KELLIA) | Data yes; don’t copy proprietary UI assets casually | Use TEI dump, not HTML scrape | **High** for data license |
| **DIOSKOROS** (dioskoros.org) full attestation DB / UI | Described as **open-access**; Impressum only found — **no clear redistribution license** on legal-notice page. Project states P6/P7 raw data will be on GitHub + APIs; treat **unpublished dump license as unknown**. | **Uncertain** for bulk dump/scrape | **setup-download** only after explicit license on GitHub dump; otherwise **citation/API use** + **manual-import** of CCL | Scraping 200k quotations without license = blocker. Prefer CCL TEI for L-evidence. **Low** for DIOSKOROS dump rights; **High** that open-access ≠ redistributable |
| **Crum, A Coptic Dictionary (1939)** | OUP © asserted on reprints; **not safely PD in US** for wholesale digital reuse | **No** (assume) | Reference-only / **manual-import** of licensed excerpt | Do not bundle OCR. **Medium–High** caution |

---

## 6. Greek NT / LXX redistributable editions

| Edition | License status | Redistributable | Setup | Notes |
| --- | --- | --- | --- | --- |
| **Westcott–Hort 1881** (PD Robinson transcription) | 1881 print PD; Robinson digital transcription treated as **public domain** for project use | **Yes** | **setup-download** (+ bundled PD pilot excerpt) | **Default Greek NT** (JAK-15 Done). Pinned at ingest (JAK-24): `byztxt/greektext-westcott-hort` commit `91473892…` + `textonly/JOH.WH` checksum in [`data/manifests/westcott-hort.json`](../data/manifests/westcott-hort.json). **High** for 1881 print PD |
| **SBLGNT** (Holmes) | **CC BY 4.0** since GitHub v1.1 (2022-12); sblgnt.com/license is CC BY 4.0; Faithlife/SBLGNT repo | **Yes** | not default | Eligible for revisit vs WH under JAK-25. Do **not** flip the default in Phase 0. Attribute SBL + Logos/Faithlife, Holmes if used. **High** for CC BY 4.0 |
| **MorphGNT: SBLGNT Edition** | Text: SBLGNT CC BY 4.0; **morphology/lemmas: CC BY-SA 3.0** | **Yes** with dual attribution; adaptations of morph layer are **ShareAlike** | optional if SBLGNT adopted later | Do not drop SA on derived morph DBs. **High** |
| **NA28 / UBS GNT** | © Deutsche Bibelgesellschaft; written permission required for redistribution | **No** | citation / short fair-use quotation policy only; not corpus | **High** |
| **Rahlfs-Hanhart LXX (2006)** | © DBG (“Septuaginta, © 2006”) | **No** | — | **High** |
| **Rahlfs LXX 1935** | Widely asserted PD in EU/Canada; **US status debated** (1935 publication; no clear original-publisher dedication found in this audit). Downstream digitisations (e.g. eliranwong) often label PD — that is **not** independent proof. | **Uncertain** for US-centric redistribution | **Not preferred** (JAK-18 Done) | **Medium** (EU PD claim common); **Low** for blanket US PD |
| **Swete LXX** via **First1KGreek / Open Greek and Latin** (also nathans/lxx-swete) | Digitization **CC BY-SA 4.0**; 19th-c. print text PD | **Yes** under BY-SA for the digital text | **bundled** / **setup-download** | **Default LXX** (JAK-18 Done). **High** for First1KGreek license |
| **OpenScriptorium Rahlfs morphology** | Claims data **CC BY 4.0**; asserts Rahlfs 1935 PD | Morphology yes if using their dump; text status still separate | **setup-download** | Treat as third-party scholarly morph, not DBG. **Medium** |

---

## 7. Public-domain English (WEB)

| Field | Finding |
| --- | --- |
| **License status** | **Public domain** dedication (eBible.org / WorldEnglish.Bible). Name **“World English Bible” is a trademark** — modified text must not be called WEB. |
| **Redistributable** | **Yes** |
| **Attribution** | Not required by copyright; courtesy credit + trademark compliance recommended. |
| **Setup mode** | **bundled** for cited parallel passages |
| **Blocking risks** | Trademark misuse if text altered. Ensure edition matches cited Greek verse mapping. SCRIPTORIUM already embeds WEB in some corpora — keep WEB PD separate from Coptic license layers. |
| **Confidence** | **High** |

**Primary refs:** https://ebible.org/Scriptures/details.php?id=engwebu ; https://www.ebible.org/legal.php ; WEB FAQ.

---

## 8. Provisional code + data licensing strategy

### Dual approach (implemented in Phase 0)

1. **Code / tooling** (`python/`, `web/`, scripts, schemas authored by the project): **Apache-2.0** (JAK-13 Done).
   - Do **not** put the whole repository under a single CC license.

2. **Data**: **No umbrella data license.** Use:
   - Root `LICENSE` = code license only
   - Optional short `NOTICE` + machine-readable **`data/manifests/*`** with: `id`, `title`, `version`, `url`, `spdx_or_custom`, `attribution`, `mode` (`bundled`|`setup-download`|`manual-import`), `checksum`, `notes`
   - Per-tree `LICENSE` / `COPYING` files next to vendored data when present
   - SPDX `REUSE.toml` optional later

3. **ShareAlike handling (JAK-16 Done):**
   - Apply SA **only** when published outputs are legal **adaptations** of BY-SA material (e.g. MorphGNT morph BY-SA 3.0, CCL / sahidic.ot / Swete BY-SA 4.0).
   - Do **not** blanket-SA the whole project, all learned stats, or independent critical texts.
   - Keep separable: raw imports, adapted tables/DBs, independent critical Greek/English, and Apache-2.0 code. Manifest each published artifact.

4. **Project-authored critical Greek + rough English:** **CC BY 4.0** (JAK-17 Done). Must contain no pasted copyrighted critical editions.

5. **Generated artifacts:** Document training corpus IDs and whether model weights are published. Prefer Horner over Wells for published training stacks.

6. **README honesty:** “code Apache-2.0; data under per-source licenses — see manifests,” never “Apache/MIT everything.”

### Default setup matrix (adopted)

| Family | Mode |
| --- | --- |
| Dilley Thomas (SCRIPTORIUM) | bundled |
| P.Oxy. via DCLP, PD Grenfell/Hunt, or Evelyn-White 1920 | bundled (small) |
| **Horner / CopSahHorner** | setup-download (default Sahidic NT) |
| **Westcott–Hort 1881** (PD Robinson) | bundled or setup-download (default Greek NT) |
| SBLGNT (+ MorphGNT) | not default; revisit JAK-25 |
| Swete LXX (First1KGreek / nathans/lxx-swete) | setup-download (default LXX) |
| sahidic.ot | setup-download / bundled |
| CCL / DDGLC TEI | setup-download / bundled (default lexicon) |
| WEB | bundled (passages) |
| Sahidica NT (+ NTAEG) / Bohairica | **excluded** from default; gated **`manual-import` only** if rights confirmed — never `setup-download` while unresolved |
| DIOSKOROS full DB | citation-only / wait for licensed dump; do not scrape (JAK-19) |
| NA28 / Rahlfs-Hanhart / Layton / Attridge / Crum OCR | not redistributed |

---

## Decisions

### Done

1. **Code license (JAK-13):** Apache-2.0 for project-owned software.
2. **Sahidic NT / Wells (JAK-14):** Horner / `CopSahHorner` default; do not bundle Sahidica, NTAEG, Bohairica without explicit rights. **Implementation/verification continues on JAK-24.**
3. **Pilot corpus (JAK-15):** Redistributable stack (Dilley + Horner + Westcott–Hort + sahidic.ot + Swete + CCL + verified Thomas fragments + WEB). Sahidica not required.
4. **ShareAlike outputs (JAK-16):** SA only on legal adaptations of BY-SA material; no blanket SA.
5. **Critical Greek/English (JAK-17):** Project-authored critical Greek + rough English = **CC BY 4.0**.
6. **LXX (JAK-18):** Default Swete via First1KGreek / nathans/lxx-swete (CC BY-SA 4.0); not Rahlfs-Hanhart; Rahlfs 1935 not preferred.
7. **Lexicon (JAK-19):** Default CCL TEI incl. DDGLC; do not wait on DIOSKOROS; no scrape.

### Follow-up (not reversing JAK-15)

8. **SBLGNT vs Westcott–Hort (JAK-25):** SBLGNT is now CC BY 4.0 (Dec 2022). Revisit whether to change the default Greek NT; **do not flip in this Phase 0 PR**.
9. **Counsel:** Optional quick IP review before claiming the repo is fully redistributable offline including all corpora.

---

## Executive risk ranking

| Priority | Issue |
| --- | --- |
| P0 | Never ship NA28 / Layton / Attridge / Rahlfs-Hanhart / Crum OCR |
| P0 | Keep Wells Sahidica / NTAEG / Bohairica out of default artifacts (`manual-import` only if rights confirmed) |
| P1 | Horner quality / verse-alignment verification at ingest (JAK-24) |
| P1 | ShareAlike only on adapted BY-SA artifacts (manifest each publish) |
| P1 | DIOSKOROS — do not scrape; use CCL |
| P2 | JAK-25 SBLGNT vs WH default revisit |
| P2 | Bohairica / Wells secondary dialect if ever revisited with rights |

**Adopted clean core stack:** Dilley Thomas (CC BY) + DCLP/PD P.Oxy. (incl. Evelyn-White 1920 where used) + **Horner/`CopSahHorner` (PD)** + **Westcott–Hort (PD Robinson)** + Swete LXX (BY-SA) + sahidic.ot (BY-SA) + CCL/DDGLC TEI (BY-SA) + WEB (PD), with Apache-2.0 code, CC BY 4.0 critical Greek/English, and per-source manifests. Wells Sahidica/NTAEG/Bohairica remain gated **`manual-import` only**.
