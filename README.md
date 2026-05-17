# GLM 5.1 Local Runtime Notes

This repository contains local orchestration scripts and experiment code for running GLM 5.1 with either the GGUF/llama.cpp path or the SGLang/KTransformers path.

## Clone Setup

```bash
git clone --recurse-submodules <repo-url>
cd glm51
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The `llama.cpp` checkout is pinned to upstream `ggml-org/llama.cpp`. A local host-memory MoE option used by `run_glm51_server.sh` is preserved as `patches/llama-host-moe.patch`:

```bash
git -C llama.cpp apply ../patches/llama-host-moe.patch
```

## NPM Malware Scanner

The npm scanner lives under `experiments/npm_malware/`. It inspects npm packages, source repositories, and package archives with a local GLM 5.1 OpenAI-compatible model endpoint. The default static analysis path does not execute package code: it copies or extracts the target into a bounded temp workspace, skips symlinks and special files, redacts likely secrets, extracts deterministic host-side indicators, and sends sanitized file excerpts to the model for JSON classification.

Main entry points:

```bash
./experiments/npm_malware/npm_malware_experiment.py
./experiments/npm_malware/npm_malware_api.py
```

Useful local commands:

```bash
# Scan a local repo or npm archive with the hardened prompt profile.
./experiments/npm_malware/npm_malware_experiment.py analyze /path/to/repo-or-archive.tgz \
  --profile hardened_numbered \
  --max-ingest-members 2000 \
  --max-total-bytes 25000000 \
  --output results/npm_repo_scan.json

# Build an offline evidence bundle, then analyze the bundle later.
./experiments/npm_malware/npm_malware_experiment.py bundle /path/to/repo-or-archive.tgz \
  --output results/npm_repo.bundle.json

./experiments/npm_malware/npm_malware_experiment.py analyze-bundle results/npm_repo.bundle.json \
  --profile hardened_numbered \
  --output results/npm_repo_scan.json

# Run synthetic red-team fixtures against the scanner.
./experiments/npm_malware/npm_malware_experiment.py redteam \
  --output results/npm_malware_redteam.json
```

The scanner can also render curated repo directories, executor specs, isolated Docker worker jobs, sinkhole-backed capture summaries, and multi-stage review handoff bundles. See `experiments/npm_malware/README.md` for the full malware-lab workflow.

## Containerized Scan API

The Docker path starts the GLM 5.1 GGUF model server inside the container and exposes only the scan API on port `8081`; the raw model server stays internal. Model shards and the persistent SQLite scan cache are stored under the host cache directory, `/home/npm_scan_models` by default.

Prewarm the GGUF model cache:

```bash
export MALWARE_SCAN_API_TOKEN='<strong-random-token>'
docker compose -f compose.nvidia-malware-scan.yml build
docker compose -f compose.nvidia-malware-scan.yml run --rm glm51-malware-scan prewarm-model
```

Start the API:

```bash
docker compose -f compose.nvidia-malware-scan.yml up -d
curl -fsS http://127.0.0.1:8081/healthz
```

Authenticated scan request:

```bash
curl -fsS http://127.0.0.1:8081/v1/scan-repo \
  -H "Authorization: Bearer $MALWARE_SCAN_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "provider": "github",
        "repo": "axios/axios",
        "branch": "main"
      }'
```

The `/v1/scan-repo` endpoint is poll-based. Active work returns `202` with a `Retry-After` header; terminal scans return `200` with result codes such as `malware_detected`, `complete_clean`, `complete_suspicious`, or `complete_review`. The request key is `provider + repo + branch/release + scan settings`, so repeating the same request polls the same logical job.

Inventory declared backend API references without running the model:

```bash
curl -fsS http://127.0.0.1:8081/v1/inventory-repo-apis \
  -H "Authorization: Bearer $MALWARE_SCAN_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "provider": "github",
        "repo": "axios/axios",
        "branch": "v1.x"
      }'
```

Operational notes:

- `MALWARE_SCAN_API_TOKEN` enables shared bearer-token auth; for multiple callers, prefer `MALWARE_SCAN_API_CLIENTS_PATH=/models/api-clients.json` with per-caller token records in the mounted model cache.
- `/healthz` is public and intentionally minimal; `/v1/admin/healthz`, `/monitor`, and `/v1/admin/monitor` require scanner auth.
- Completed identical scans are served from persistent job state and cache when the repo fingerprint still matches. Changed repo fingerprints trigger a fresh scan.
- Uncached model work is admission-controlled with `SCAN_MAX_ACTIVE_UNCACHED`, `SCAN_MAX_QUEUED_UNCACHED`, and `SCAN_UNCACHED_ACQUIRE_TIMEOUT_S`.
- The API accepts GitHub repo scans with exactly one of `branch` or `release`; source URLs with credentials, private or loopback hosts, fragments, and unsafe redirects are rejected by default.

## Secrets and Local Artifacts

Do not commit tokens, local `.env` files, model weights, cache databases, runtime logs, or generated benchmark/result output. Those paths are ignored by the root `.gitignore`; pass credentials through environment variables at runtime.
