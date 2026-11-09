# Local downloads (gitignored)

Place explicit setup downloads here. Contents of this directory (except this README) are **not** committed.

## Download (preferred)

From the repo root (requires network once):

```bash
uv run thomas sources download
# equivalent: uv run python scripts/download_pilot_sources.py
```

That command:

1. Fetches CrossWire `CopSahHorner.zip`, verifies the sha256 from `data/manifests/copsahhorner.json`, and unpacks to `data/downloads/sword/`
2. Fetches INTF/NTVMR Codex Vaticanus GA 03 (`docID=20003`) **teiraw** for the pilot ranges into `data/downloads/ntvmr-vaticanus/`, verifying sha256 pins from `data/manifests/ntvmr-vaticanus-b.json` (throttled; **no manuscript images**)

Re-fetch even when local files already match:

```bash
uv run thomas sources download --force
```

## Regenerate pilot (optional)

After a successful download:

```bash
uv sync --locked --extra sword   # or rely on the dev group
uv run thomas sources build-pilot
```

## Offline validate

Offline validation does **not** need this directory:

```bash
uv run thomas sources validate
```
