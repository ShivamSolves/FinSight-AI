"""
conftest.py — Makes the project root importable on Streamlit Cloud and in tests.

Streamlit Cloud runs app.py from /mount/src/finsight-ai/app/app.py,
so the project root (/mount/src/finsight-ai) is not automatically on sys.path.
This file sits at the project root and is auto-loaded by Python, inserting
the root into sys.path so `import finsight` always works.
"""
import sys
from pathlib import Path

# Insert project root at the front of sys.path
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
