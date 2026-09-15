"""
Smoke test: run every section of the dashboard and fail on any exception.

The app has ten sections, dozens of widgets and about forty charts. A typo in a
column name only shows up when that specific tab renders, so this walks all of
them headlessly.

Run:  python tests/test_dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
SECTIONS = ["Overview", "Respondents", "Employer support", "Stigma and openness",
            "Treatment drivers", "Modelling", "Geography", "Explorer", "Findings",
            "Data quality"]


def run_section(name: str, timeout: int = 120) -> tuple[bool, list[str]]:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    if at.exception:
        return False, [str(e.message) for e in at.exception]
    at.radio[0].set_value(name).run()
    if at.exception:
        return False, [str(e.message) for e in at.exception]

    # Exercise every tab in the section, since Streamlit renders all tab bodies
    # on each run but interactive widgets inside them only resolve when set.
    for widget in list(at.selectbox) + list(at.radio)[1:]:
        try:
            options = widget.options
            if len(options) > 1:
                widget.set_value(options[-1]).run()
                if at.exception:
                    return False, [f"{name}/{widget.label}: {e.message}" for e in at.exception]
        except Exception as exc:  # widget not settable in this state
            return False, [f"{name}/{getattr(widget, 'label', '?')}: {exc}"]
    return True, []


def main() -> int:
    failures = []
    for name in SECTIONS:
        start = time.time()
        ok, errors = run_section(name)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:24s} {time.time() - start:5.1f}s")
        if not ok:
            failures.append((name, errors))
            for err in errors[:3]:
                print(f"         {err.splitlines()[0][:160]}")

    print()
    if failures:
        print(f"{len(failures)} of {len(SECTIONS)} sections failed.")
        return 1
    print(f"All {len(SECTIONS)} sections rendered without error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
