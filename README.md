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

## Secrets and Local Artifacts

Do not commit tokens, local `.env` files, model weights, cache databases, runtime logs, or generated benchmark/result output. Those paths are ignored by the root `.gitignore`; pass credentials through environment variables at runtime.
