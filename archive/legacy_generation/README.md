# Legacy direct generation path

This directory preserves Lumen's original direct retrieval-to-Ollama answer
generator and its batch harness for historical provenance.

These files are not part of the active application. They were superseded by:

- `src/api/app.py` for the HTTP application;
- `src/agents/graph.py` for routing, synthesis, verification, and review;
- `src/agents/run_graph.py` for the command-line graph runner; and
- `src/llm/local_client.py` for all role-based Ollama calls.

`src/generation/lab_query.py` remains active because the graph uses its
structured laboratory-data retrieval. The archived files may refer to their
former `src.generation` module paths and are retained as historical source, not
as supported commands.
