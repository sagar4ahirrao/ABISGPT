"""Non-interactive SSH helper for the remote GPU box.

Usage:
    python scripts/remote_exec.py probe                    # environment probe
    python scripts/remote_exec.py run "nvidia-smi"         # exec a command
    python scripts/remote_exec.py put local_path remote_path
    python scripts/remote_exec.py get remote_path local_path

Password is read from env ABISGPT_SSH_PW (set by the caller), never hardcoded.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

HOST = "13.235.199.152"
PORT = 2222
USER = "development"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._ssh_session import connect  # noqa: E402


def sh(cmd: str) -> tuple[int, str]:
    """Exec a command remotely; full stdout+stderr. KMP dup ok."""
    wrapped = (f"export KMP_DUPLICATE_LIB_OK=TRUE; {cmd}")
    client = connect()
    try:
        _, out, err = client.exec_command(wrapped, timeout=600)
        ret = out.channel.recv_exit_status()
        txt = out.read().decode("utf-8", "replace") + err.read().decode("utf-8", "replace")
        return ret, txt
    finally:
        client.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        ret, txt = sh(textwrap.dedent(
            """
            echo "== who/where =="; hostname; whoami; nproc; free -g | head -2
            echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>&1 | head -3
            echo "== python =="; python3 --version 2>&1; which python3
            echo "== torch =="; python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tail -1
            echo "== libs =="; python3 -c "import transformers; print('transformers', transformers.__version__)" 2>&1 | tail -1
            python3 -c "import cv2; print('cv2', cv2.__version__)" 2>&1 | tail -1
            python3 -c "import numpy; print('numpy', numpy.__version__)" 2>&1 | tail -1
            echo "== disk/workdir =="; df -h . | tail -1; pwd; ls -la | head -5
            """))
        print(f"(exit {ret})")
        print(txt)
    elif mode == "run":
        ret, txt = sh(sys.argv[2])
        print(f"(exit {ret})")
        print(txt)
    elif mode in ("put", "get"):
        client = connect()
        try:
            sftp = client.open_sftp()
            if mode == "put":
                sftp.put(sys.argv[2], sys.argv[3])
                print(f"put {sys.argv[2]} -> {sys.argv[3]}")
            else:
                sftp.get(sys.argv[2], sys.argv[3])
                print(f"got {sys.argv[2]} -> {sys.argv[3]}")
        finally:
            client.close()
    else:
        print("unknown mode")
