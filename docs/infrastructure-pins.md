# Infrastructure pins

The container references below were resolved from their registries on
2026-09-22 and are pinned by both readable release tag and multi-platform
manifest digest:

| Component | Pin |
|---|---|
| API base | `python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e` |
| Postgres + pgvector | `pgvector/pgvector:0.8.6-pg16-bookworm@sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b` |
| Demo Ollama | `ollama/ollama:0.31.1@sha256:f1a705f2bd113fb8d15f85f7c217f0dc5f6bebda6b0cc42b82c3ad165ffcb9dc` |

The RunPod bootstrap compiles the same pgvector 0.8.6 tag and asks Ollama's
official installer for version 0.31.1, then verifies the reported version.

To update an image, inspect the exact candidate with
`docker buildx imagetools inspect`, review its upstream release notes, update
the tag and manifest-list digest together, render both Compose files, and run
the full test/smoke matrix. Do not replace these with `latest` or a moving major
tag.

Remaining variability is explicit: Debian package mirrors and the RunPod base
template are external mutable inputs. The Pod reuses its CUDA-matched PyTorch;
the final-evaluation manifest records the exact OS, Python, torch, CUDA, Ollama,
model, and database-extension versions so a run cannot imply stronger
reproducibility than the platform provides.
