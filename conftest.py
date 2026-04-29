"""
SENTRY — Pytest Configuration
================================
Adds the project root to sys.path so all test files can import
project modules (ai_engine, middleware, config, etc.) without
needing relative import hacks.

This file is automatically discovered by pytest and also applies
when running test files directly with: python tests/unit/test_xyz.py
"""

import sys
from pathlib import Path

# Insert project root at the front of the path
sys.path.insert(0, str(Path(__file__).resolve().parent))
