import os
import sys
from pathlib import Path

# API tests run against the demo plane with tracing off; nothing here touches
# Postgres, Ollama or the models — every dependency is replaced per test.
os.environ["LUMEN_DATA_PLANE"] = "demo"
os.environ["LUMEN_TRACING"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
