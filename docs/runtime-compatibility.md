# Runtime compatibility

Lumen's canonical interpreter is CPython 3.12. The checked-in direct
dependencies all declare support for 3.12, including the installed PyTorch
2.11, Transformers 5.8, NumPy 2.4, pandas 3.0, Presidio, spaCy 3.8, LangGraph,
FastAPI, and psycopg stacks. The repository's local validation environment may
be newer, but final/cloud readiness requires the canonical minor version.

Why 3.12: it is supported by every pinned package, is the upper end of
PyTorch's documented macOS recommendation, and is available in current NVIDIA
Python/CUDA tooling. It avoids making the project depend on newer interpreter
support merely because one laptop happens to have it.

The laptop and Docker image install `torch==2.11.0` from `requirements.txt`.
The RunPod bootstrap intentionally reuses the template's CUDA-enabled PyTorch
instead of replacing its driver-matched binary; it requires Python 3.12 and
records the exact torch/CUDA versions in evaluation provenance. Select a
PyTorch template whose preinstalled torch is compatible with the Pod driver.

`sentence-transformers` is not a dependency: Lumen loads Transformers models
directly. PyYAML is not required by the active runtime; the remaining YAML file
is archived research provenance.
