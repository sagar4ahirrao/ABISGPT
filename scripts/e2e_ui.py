"""Real-browser (Chromium) end-to-end test of the ABISGPT console.

Drives http://localhost:8000 with Playwright exactly like a user: clicks the
drop zone, picks a real face image from the OS file chooser, then exercises
every action. Console errors and page exceptions are captured as evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"
SHOTDIR = Path("artifacts/ui_shots")


def make_face_jpg(path: Path, subject: int = 0) -> None:
    """A realistic 256x256 jpg from the Olivetti set (real face texture)."""
    z = np.load("artifacts/dataset/olivetti.npz")
    frames, labels = z["frames"], z["labels"].astype(int)
    f = frames[np.flatnonzero(labels == subject)[0]]
    img = (f * 255).astype(np.uint8)
    big = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)  # HxWx3 RGB
    cv2.imwrite(str(path), big[..., ::-1])  # cv2 wants BGR


def main() -> int:
    SHOTDIR.mkdir(parents=True, exist_ok=True)
    face = Path("artifacts/ui_face_a.jpg").resolve()
    make_face_jpg(face, subject=0)
    other = Path("artifacts/ui_face_b.jpg").resolve()
    make_face_jpg(other, subject=17)

    failures: list[str] = []
    console_errors: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{'PASS' if cond else 'FAIL'}  {name}{(' :: ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    import requests

    def reset_registry() -> None:
        ids = requests.get(f"{BASE}/identities", timeout=10).json()["identities"]
        for i in list(ids):
            if i.startswith("ui_"):
                requests.delete(f"{BASE}/identities/{i}", timeout=10)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 1000})
        page.on("console", lambda m: console_errors.append(
            m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        # ---- 1. page loads -------------------------------------------------
        page.goto(BASE, timeout=20000)
        page.wait_for_timeout(1500)
        check("page title", "ABISGPT Biometric Console" in page.title(),
              page.title())
        check("upload zone visible", page.is_visible("#drop"))
        page.screenshot(path=str(SHOTDIR / "01_loaded.png"))

        # ---- 2. THE UPLOAD FEATURE (user-reported broken) ------------------
        with page.expect_file_chooser() as fc:
            page.click("#drop")
        chooser = fc.value
        chooser.set_files(face)
        page.wait_for_selector("#preview", state="visible", timeout=5000)
        meta = page.text_content("#meta") or ""
        check("upload sets preview", page.is_visible("#preview"), meta.strip())
        check("upload enables Run",
              page.eval_on_selector("#run", "el=>!el.disabled"))
        page.screenshot(path=str(SHOTDIR / "02_uploaded.png"))

        # ---- 3. enroll the uploaded face -----------------------------------
        page.fill("#identity", "ui_alice")
        page.select_option("#mode", "enroll")
        page.click("#run")
        ok = False
        for _ in range(30):                     # poll up to 15 s
            page.wait_for_timeout(500)
            if "ENROLLED" in (page.text_content("#result") or ""):
                ok = True
                break
        detail = (page.text_content("#result") or "").strip()
        check("enroll via UI", ok, detail[:120] +
              " | console: " + " / ".join(console_errors[-3:]))
        page.screenshot(path=str(SHOTDIR / "03_enrolled.png"))

        # ---- 4. verify 1:1 via UI -------------------------------------------
        page.select_option("#mode", "verify")
        page.wait_for_timeout(400)
        opts = page.eval_on_selector_all("#identity-sel option",
                                         "els=>els.map(e=>e.textContent)")
        check("identity dropdown populated", "ui_alice" in opts, f"{opts}")
        try:
            page.select_option("#identity-sel", "ui_alice")
        except Exception:  # noqa: BLE001
            opts = page.eval_on_selector_all(
                "#identity-sel option", "els=>els.map(e=>e.textContent)")
            check("ui_alice selectable", False, f"options={opts}")
        page.click("#run")
        page.wait_for_timeout(4000)
        txt = (page.text_content("#result") or "").strip()
        check("verify via UI says MATCH", "MATCH" in txt, txt[:160])
        page.screenshot(path=str(SHOTDIR / "04_verify.png"))

        # ---- 5. identify 1:N via UI ------------------------------------------
        page.select_option("#mode", "identify")
        page.click("#run")
        page.wait_for_timeout(4000)
        txt = (page.text_content("#result") or "").strip()
        check("identify via UI ranks ui_alice", "ui_alice" in txt, txt[:160])
        page.screenshot(path=str(SHOTDIR / "05_identify.png"))

        # ---- 6. PAD-only via UI ------------------------------------------------
        page.select_option("#mode", "pad")
        page.click("#run")
        page.wait_for_timeout(4000)
        txt = (page.text_content("#result") or "").strip()
        check("pad via UI gives verdict",
              any(t in txt for t in ("LIVE", "NOT LIVE")), txt[:160])
        page.screenshot(path=str(SHOTDIR / "06_pad.png"))

        # ---- 7. register a second person + crossmatch ---------------------
        with page.expect_file_chooser() as fc2:
            page.click("#drop")
        fc2.value.set_files(other)
        page.wait_for_timeout(800)
        page.select_option("#mode", "enroll")
        page.fill("#identity", "ui_bob")
        page.click("#run")
        for _ in range(12):
            page.wait_for_timeout(500)
            if "ENROLLED" in (page.text_content("#result") or ""):
                break
        page.select_option("#mode", "crossmatch")
        page.click("#run")
        page.wait_for_timeout(3000)
        txt = (page.text_content("#result") or "").strip()
        check("crossmatch via UI runs",
              "2 ids" in txt or "duplicate" in txt or "clean" in txt, txt[:160])
        page.screenshot(path=str(SHOTDIR / "07_crossmatch.png"))

        # ---- 8. registry + forget --------------------------------------------
        page.click("#mode-hint", click_count=0)  # no-op, keep focus stable
        rows = page.eval_on_selector_all("#reg tr", "els=>els.length")
        check("registry lists 2 identities", rows >= 2, f"rows={rows}")
        page.screenshot(path=str(SHOTDIR / "08_registry.png"))

        browser.close()

    # reset live registry even if the browser part failed
    reset_registry()

    check("no console/page errors", not console_errors,
          " | ".join(console_errors)[:200])
    print("\n==", "ALL PASS" if not failures else f"FAILURES: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
