"""CLI wrapper for the SENTRY knowledge-base ingestor.

The ingestion implementation lives in ``ai_engine.ingestion`` so it can be
tested and imported like normal application code. This wrapper preserves the
documented command:

    python scripts/ingest_knowledge_base.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.ingestion.knowledge_base_ingestor import main  # noqa: E402


if __name__ == "__main__":
    main()
