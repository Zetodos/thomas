# Logion 7 human review (model-v3)

This page is a human-review copy of the committed **model-v3** output for Gospel of Thomas logion 7 (`thomas.7`). It is a new page for that output. The JAK-40 freeze and its `review_status` stay untouched. The JAK-44 review read the older package at `data/reconstructions/jak40/`.

No model was retrained. The prediction files were not edited. No P.Oxy. reading and no other fragment Greek was used to fill, replace, or drop a token on this page.

## Sources and hashes

| Item | Value |
| --- | --- |
| Greek output | `data/generated/thomas-generated-v3.json` |
| Output file SHA-256 | `6c358d57d8d18a7613a616bb69c29d0ef1f8f39592e333bc31e665e9646eb54c` |
| `thomas.7` object SHA-256 | `403d2489e5510c097c4dbd164f7cb4c0c76671b42f1172b0b306c27dedf68e88` |
| Coptic input | `data/sources/thomas/dilley-thomas-v6.2.0.json`, unit `thomas.7` |
| Coptic file SHA-256 | `34b24f4fcec8daa3540fca451d5a8a6d8d5fa02b4c4b803537c8b08b2e9fc501` |
| Commit of these files | `c28c656c7c890a36ce629067344d0cb0dfa590d2` on `origin/main` |
| `model_id` | `model-v3` |
| `coptic_source_id` | `dilley-thomas` |
| `unit_id` | `thomas.7` |

The output-file hash is SHA-256 over the raw bytes of `data/generated/thomas-generated-v3.json`. The object hash is SHA-256 over the UTF-8 bytes of the `thomas.7` JSON object as it sits in that file: from the `{` that opens the object containing `"unit_id":"thomas.7"` through the matching `}`. Braces inside JSON strings are not counted. This page does not change either source file.

The committed unit has `unit_id`, `coptic_source_id`, `greek_text`, and `tokens`. It has no `coptic_text`, no `clause_alternatives`, and no per-token `alternatives`. The reconstruct writer drops `coptic_text` when the corpus file must stay under 1024 KiB. The decoder sets `coptic_text` from the Dilley unit's `normalized_text` before that drop. The Coptic below is that unit, copied from the committed Dilley file. In that file, `normalized_text` is the morpheme `norm` fields joined by single spaces, in list order. The file omits `seq` on each morpheme. The list index is the index stored in `coptic_seqs`.

## What model-v3 emitted

model-v3 stores one Greek sentence for this unit, plus the token list below. The committed unit has no span records. The fenced block in the Greek sentence section is `greek_text`. It is the token `surface` values joined by single spaces, in `seq` order.

Each Greek token stores one Coptic index in `coptic_seqs`. In `NeuralKoineDecoder._decode_chunk` (`python/zetodos_thomas/neural/decoder_v2.py`), that index is the top-1 cross-attention mass for the word. If that mass is empty, the decoder still stores one monotonic fallback index inside the chunk. On this unit every token has exactly one index. Twenty-seven Greek tokens can name at most 27 of the 42 morphemes. On this unit several tokens share an index, so 23 morphemes have no pointer. The file does not store a null for those morphemes.

A dash in the Greek column means no stored token lists that index. The reading column sorts those 23 gaps. They are not one class:

- **absorbed** — an analytic morpheme in a bound group whose inflected Greek word is in the sentence. Sahidic splits that word across the group (`ⲡ`+`ⲙⲟⲩⲓ`, `ⲛⲁ`+`ⲟⲩⲟⲙ`+`ϥ`, `ⲡⲁⲓ`+`ⲉⲧⲉ`, `ⲛϭⲓ`+`ⲡ`+`ⲣⲱⲙⲉ`, and the copula `ⲡⲉ` beside `ⲙⲁⲕⲁⲣⲓⲟⲥ`). The single pointer sits on one member of the group.
- **pointer drift** — the Greek word is in the sentence, and its single top-1 index names a different morpheme. Token 0 `λέγει` lists index 3 `ⲙⲁⲕⲁⲣⲓⲟⲥ`, so index 0 `ⲡⲉϫⲉ` has no pointer.
- **omitted** — index 41 `ⲣⲱⲙⲉ` only. No Greek token remains for that noun. The sentence ends at `ἔσται`.
- Index 24 is the Dilley dot. It is punctuation, not the omitted noun.

Token order is not Coptic order, so the Greek column is not a second sentence. Reading it from top to bottom does not rebuild `greek_text`. The pointers below are copied as stored.

## Visible defects in the stored alignment

The Greek text is the committed sentence in the Greek sentence section. These four mappings are what that sentence and its `coptic_seqs` show:

| What is visible | Stored alignment |
| --- | --- |
| Final `ⲣⲱⲙⲉ` omitted | Index 41 has no Greek token. The sentence ends `καί ὁ λέων ἔσται`. |
| `ⲃⲏⲧ` → `ἐβλασφήμησεν` | Index 23 is pointed by token 16 `ἐβλασφήμησεν` (and token 15 `καί`). |
| `ϣⲱⲡⲉ ⲛ ⲣⲱⲙⲉ` → `τῶν ἀνθρώπων` | Index 18 `ϣⲱⲡⲉ` is pointed by `ἔσται` (10), index 19 `ⲛ` by `τῶν` (13), index 20 `ⲣⲱⲙⲉ` by `ἀνθρώπων` (14). |
| `λέγει` → `ⲙⲁⲕⲁⲣⲓⲟⲥ` | Token 0 `λέγει` lists index 3 `ⲙⲁⲕⲁⲣⲓⲟⲥ`. Index 0 `ⲡⲉϫⲉ` has no pointer. |

`provenance` is the stored code on each token (`L` loanword, `N` NT bitext, as on the generated-token schema). `confidence` is the stored number.

`data/generated/eval-report-v3.json` also stores a shorter `predicted_greek` for span `thomas.7.poxy654.a`. That span covers Coptic indexes 2 through 41 (`coptic_start` 2, `coptic_end` 42 in `data/evidence/fragment-clause-map.json`). A token is kept in that span string only when its `coptic_seqs` meet that range, so the span string is a scoring slice of this unit. This page copies the unit sentence. It leaves the span string, the papyrus ink, and any editorial restoration in the eval report.

## Coptic input

Manuscript-oriented `original_text`:

```
ⲡⲉϫⲉⲓ︤ⲥ︥ ⲟⲩⲙⲁⲕⲁⲣⲓⲟⲥ ⲡⲉ ⲡⲙⲟⲩⲉⲓ ⲡⲁⲉⲓ ⲉⲧⲉⲡⲣⲱⲙⲉ ⲛⲁⲟⲩⲟⲙϥ ⲁⲩⲱ ⲛ̄ⲧⲉⲡⲙⲟⲩⲉⲓ ϣⲱⲡⲉ ⲣ̄ⲣⲱⲙⲉ ⲁⲩⲱ ϥⲃⲏⲧ · ⲛ̄ϭⲓⲡⲣⲱⲙⲉ ⲡⲁⲉⲓ ⲉⲧⲉⲡⲙⲟⲩⲉⲓ ⲛⲁⲟⲩⲟⲙϥ ⲁⲩⲱ ⲡⲙⲟⲩⲉⲓ ⲛⲁϣⲱⲡⲉ ⲣ̄ⲣⲱⲙⲉ
```

Analysis `normalized_text` (42 morphemes; this is the string the decoder copies into `coptic_text`):

```
ⲡⲉϫⲉ ⲓⲏⲥⲟⲩⲥ ⲟⲩ ⲙⲁⲕⲁⲣⲓⲟⲥ ⲡⲉ ⲡ ⲙⲟⲩⲓ ⲡⲁⲓ ⲉⲧⲉ ⲡ ⲣⲱⲙⲉ ⲛⲁ ⲟⲩⲟⲙ ϥ ⲁⲩⲱ ⲛⲧⲉ ⲡ ⲙⲟⲩⲓ ϣⲱⲡⲉ ⲛ ⲣⲱⲙⲉ ⲁⲩⲱ ϥ ⲃⲏⲧ · ⲛϭⲓ ⲡ ⲣⲱⲙⲉ ⲡⲁⲓ ⲉⲧⲉ ⲡ ⲙⲟⲩⲓ ⲛⲁ ⲟⲩⲟⲙ ϥ ⲁⲩⲱ ⲡ ⲙⲟⲩⲓ ⲛⲁ ϣⲱⲡⲉ ⲣ ⲣⲱⲙⲉ
```

## Greek sentence

```
λέγει ὁ Ἰησοῦς μακάριος ὁ λέων ὅς φάγεται ἄνθρωπος καί ἔσται ὁ λέων τῶν ἀνθρώπων καί ἐβλασφήμησεν ὁ ἄνθρωπος ὅν φάγεται ὁ λέων καί ὁ λέων ἔσται
```

27 tokens.

## Alignment

| Coptic seq | norm | lemma | pos | stored Greek pointer | reading |
| ---: | --- | --- | --- | --- | --- |
| 0 | ⲡⲉϫⲉ | ⲡⲉϫⲉ | VBD | — | pointer drift: `λέγει` (0) points at 3 |
| 1 | ⲓⲏⲥⲟⲩⲥ | ⲓⲏⲥⲟⲩⲥ | NPROP | ὁ (1); Ἰησοῦς (2) | pointed |
| 2 | ⲟⲩ | ⲟⲩ | ART | — | absorbed: article of `ⲙⲁⲕⲁⲣⲓⲟⲥ` (3) |
| 3 | ⲙⲁⲕⲁⲣⲓⲟⲥ | ⲙⲁⲕⲁⲣⲓⲟⲥ | N | λέγει (0); μακάριος (3) | pointed |
| 4 | ⲡⲉ | ⲡⲉ | COP | — | absorbed: copula beside `μακάριος` (3) |
| 5 | ⲡ | ⲡ | ART | — | absorbed: article of `ⲙⲟⲩⲓ` (6) |
| 6 | ⲙⲟⲩⲓ | ⲙⲟⲩⲓ | N | ὁ (4); λέων (5) | pointed |
| 7 | ⲡⲁⲓ | ⲡⲁⲓ | PDEM | ὅς (6) | pointed |
| 8 | ⲉⲧⲉ | ⲉⲧⲉⲣⲉ | CREL | — | absorbed: relative marker of `ⲡⲁⲓ` (7) |
| 9 | ⲡ | ⲡ | ART | — | absorbed: article of `ⲣⲱⲙⲉ` (10) |
| 10 | ⲣⲱⲙⲉ | ⲣⲱⲙⲉ | N | — | pointer drift: `ἄνθρωπος` (8) points at 14 |
| 11 | ⲛⲁ | ⲛⲁ | FUT | — | absorbed: future of `φάγεται` (7) on 12 |
| 12 | ⲟⲩⲟⲙ | ⲟⲩⲱⲙ | V | φάγεται (7) | pointed |
| 13 | ϥ | ⲛⲧⲟϥ | PPERO | — | absorbed: object pronoun of `φάγεται` (7) |
| 14 | ⲁⲩⲱ | ⲁⲩⲱ | CONJ | ἄνθρωπος (8); καί (9) | pointed |
| 15 | ⲛⲧⲉ | ⲛⲧⲉ | ACONJ | — | absorbed: conjunctive auxiliary of `ϣⲱⲡⲉ` (18) |
| 16 | ⲡ | ⲡ | ART | — | absorbed: article of `ⲙⲟⲩⲓ` (17) |
| 17 | ⲙⲟⲩⲓ | ⲙⲟⲩⲓ | N | ὁ (11); λέων (12) | pointed |
| 18 | ϣⲱⲡⲉ | ϣⲱⲡⲉ | V | ἔσται (10) | pointed |
| 19 | ⲛ | ⲛ | PREP | τῶν (13) | pointed |
| 20 | ⲣⲱⲙⲉ | ⲣⲱⲙⲉ | N | ἀνθρώπων (14) | pointed |
| 21 | ⲁⲩⲱ | ⲁⲩⲱ | CONJ | — | pointer drift: `καί` (15) points at 23 |
| 22 | ϥ | ⲛⲧⲟϥ | PPERS | — | absorbed: subject pronoun of `ⲃⲏⲧ` (23) |
| 23 | ⲃⲏⲧ | ⲃⲱⲧⲉ | VSTAT | καί (15); ἐβλασφήμησεν (16) | pointed |
| 24 | · | · | PUNCT | — | punctuation |
| 25 | ⲛϭⲓ | ⲛϭⲓ | PTC | — | absorbed: subject marker; `ὁ ἄνθρωπος` points at 26–27 |
| 26 | ⲡ | ⲡ | ART | ὁ (17) | pointed |
| 27 | ⲣⲱⲙⲉ | ⲣⲱⲙⲉ | N | ἄνθρωπος (18) | pointed |
| 28 | ⲡⲁⲓ | ⲡⲁⲓ | PDEM | ὅν (19) | pointed |
| 29 | ⲉⲧⲉ | ⲉⲧⲉⲣⲉ | CREL | — | absorbed: relative marker of `ⲡⲁⲓ` (28) |
| 30 | ⲡ | ⲡ | ART | — | absorbed: article of `ⲙⲟⲩⲓ` (31) |
| 31 | ⲙⲟⲩⲓ | ⲙⲟⲩⲓ | N | λέων (22) | pointed |
| 32 | ⲛⲁ | ⲛⲁ | FUT | — | absorbed: future of `φάγεται` (20) on 33 |
| 33 | ⲟⲩⲟⲙ | ⲟⲩⲱⲙ | V | φάγεται (20); ὁ (21) | pointed |
| 34 | ϥ | ⲛⲧⲟϥ | PPERO | — | absorbed: object pronoun of `φάγεται` (20) |
| 35 | ⲁⲩⲱ | ⲁⲩⲱ | CONJ | καί (23) | pointed |
| 36 | ⲡ | ⲡ | ART | — | absorbed: article of `ⲙⲟⲩⲓ` (37) |
| 37 | ⲙⲟⲩⲓ | ⲙⲟⲩⲓ | N | ὁ (24); λέων (25) | pointed |
| 38 | ⲛⲁ | ⲛⲁ | FUT | — | absorbed: future of `ϣⲱⲡⲉ` (39) |
| 39 | ϣⲱⲡⲉ | ϣⲱⲡⲉ | V | — | pointer drift: `ἔσται` (26) points at 40 |
| 40 | ⲣ | ⲛ | PREP | ἔσται (26) | pointed |
| 41 | ⲣⲱⲙⲉ | ⲣⲱⲙⲉ | N | — | omitted |

19 Coptic indexes have at least one Greek token. Of the 23 with no pointer, 17 are absorbed, 4 are pointer drift, 1 is punctuation, and 1 (index 41) is omitted.

## Greek tokens as stored

| seq | surface | lemma | morph | coptic seqs | coptic norm | provenance | confidence |
| ---: | --- | --- | --- | --- | --- | --- | ---: |
| 0 | λέγει | λέγω | pos=V- person=3 tense=P voice=A mood=I number=S | 3 | ⲙⲁⲕⲁⲣⲓⲟⲥ | N | 0.4702 |
| 1 | ὁ | ὁ | pos=RA case=N number=S gender=M | 1 | ⲓⲏⲥⲟⲩⲥ | N | 0.4702 |
| 2 | Ἰησοῦς | Ἰησοῦς | pos=N- case=N number=S gender=M | 1 | ⲓⲏⲥⲟⲩⲥ | N | 0.4702 |
| 3 | μακάριος | μακάριος | pos=A- case=N number=S gender=M | 3 | ⲙⲁⲕⲁⲣⲓⲟⲥ | L | 0.4702 |
| 4 | ὁ | ὁ | pos=RA case=N number=S gender=M | 6 | ⲙⲟⲩⲓ | N | 0.4702 |
| 5 | λέων | λέων | pos=N- case=N number=S gender=M | 6 | ⲙⲟⲩⲓ | N | 0.4702 |
| 6 | ὅς | ὅς | pos=RR case=N number=S gender=M | 7 | ⲡⲁⲓ | N | 0.4702 |
| 7 | φάγεται | ἐσθίω | pos=V- person=3 tense=F voice=M mood=I number=S | 12 | ⲟⲩⲟⲙ | N | 0.4702 |
| 8 | ἄνθρωπος | ἄνθρωπος | pos=N- case=N number=S gender=M | 14 | ⲁⲩⲱ | N | 0.4702 |
| 9 | καί | καί | pos=C- | 14 | ⲁⲩⲱ | N | 0.4702 |
| 10 | ἔσται | εἰμί | pos=V- person=3 tense=F voice=M mood=I number=S | 18 | ϣⲱⲡⲉ | N | 0.4702 |
| 11 | ὁ | ὁ | pos=RA case=N number=S gender=M | 17 | ⲙⲟⲩⲓ | N | 0.4702 |
| 12 | λέων | λέων | pos=N- case=N number=S gender=M | 17 | ⲙⲟⲩⲓ | N | 0.4702 |
| 13 | τῶν | ὁ | pos=RA case=G number=P gender=M | 19 | ⲛ | N | 0.4702 |
| 14 | ἀνθρώπων | ἄνθρωπος | pos=N- case=G number=P gender=M | 20 | ⲣⲱⲙⲉ | N | 0.4702 |
| 15 | καί | καί | pos=C- | 23 | ⲃⲏⲧ | N | 0.4702 |
| 16 | ἐβλασφήμησεν | βλασφημέω | pos=V- person=3 tense=A voice=A mood=I number=S | 23 | ⲃⲏⲧ | N | 0.4702 |
| 17 | ὁ | ὁ | pos=RA case=N number=S gender=M | 26 | ⲡ | N | 0.4702 |
| 18 | ἄνθρωπος | ἄνθρωπος | pos=N- case=N number=S gender=M | 27 | ⲣⲱⲙⲉ | N | 0.4702 |
| 19 | ὅν | ὅς | pos=RR case=A number=S gender=M | 28 | ⲡⲁⲓ | N | 0.4702 |
| 20 | φάγεται | ἐσθίω | pos=V- person=3 tense=F voice=M mood=I number=S | 33 | ⲟⲩⲟⲙ | N | 0.4702 |
| 21 | ὁ | ὁ | pos=RA case=N number=S gender=M | 33 | ⲟⲩⲟⲙ | N | 0.4702 |
| 22 | λέων | λέων | pos=N- case=N number=S gender=M | 31 | ⲙⲟⲩⲓ | N | 0.4702 |
| 23 | καί | καί | pos=C- | 35 | ⲁⲩⲱ | N | 0.1658 |
| 24 | ὁ | ὁ | pos=RA case=N number=S gender=M | 37 | ⲙⲟⲩⲓ | N | 0.1658 |
| 25 | λέων | λέων | pos=N- case=N number=S gender=M | 37 | ⲙⲟⲩⲓ | N | 0.1658 |
| 26 | ἔσται | εἰμί | pos=V- person=3 tense=F voice=M mood=I number=S | 40 | ⲣ | N | 0.1658 |
