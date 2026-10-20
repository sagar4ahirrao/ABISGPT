# Session-only ssh plumbing (password from env ABISGPT_SSH_PW, never stored).

import os

import paramiko

HOST = "13.235.199.152"
PORT = 2222
USER = "development"


def connect() -> paramiko.SSHClient:
    pw = os.environ.get("ABISGPT_SSH_PW", "")
    if not pw:
        raise SystemExit("ABISGPT_SSH_PW not set — password does not touch disk")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(HOST, port=PORT, username=USER, password=pw, timeout=30,
                  allow_agent=False, look_for_keys=False)
    except Exception:
        c.close()
        raise
    return c
