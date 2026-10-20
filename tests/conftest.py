"""pytest shared fixtures.

`KMP_DUPLICATE_LIB_OK` shields a known Windows-only clash: loading
torch and faiss (both bundle OpenMP) in one process can abort.
Lazy imports usually prevent it; this is defense in depth.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

