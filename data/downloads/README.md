# Local downloads (gitignored)

Place explicit setup downloads here. Contents of this directory (except this README) are **not** committed.

## CopSahHorner (Sahidic NT)

```bash
curl -fsSL -o data/downloads/CopSahHorner.zip \
  https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/CopSahHorner.zip
mkdir -p data/downloads/sword
unzip -o data/downloads/CopSahHorner.zip -d data/downloads/sword
# Expected: sha256:4f074891e60cab8e59cb8ebaa64cb207ae990ae78270df5be290fb0a8e7d2132
```

## Westcott–Hort (Greek NT)

Fetch Robinson `textonly/*.WH` files from
`https://github.com/byztxt/greektext-westcott-hort` (pin commit in
[`../manifests/westcott-hort.json`](../manifests/westcott-hort.json)) into
`data/downloads/westcott-hort/textonly/`.

## Regenerate pilot (optional)

```bash
uv sync --locked --extra sword   # or rely on the dev group
uv run thomas sources build-pilot
```

Offline validation does **not** need this directory:

```bash
uv run thomas sources validate
```
