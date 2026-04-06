"""
Legacy CLI stub kept only to redirect old workflows.

The supported evaluation entrypoint is `evaluate_oos.py`.
"""

from __future__ import annotations


DEPRECATION_MESSAGE = (
    "\n"
    "test_agent.py is a deprecated legacy script.\n"
    "It used RecurrentPPO, which is incompatible with the current MaskablePPO stack.\n"
    "\n"
    "Use the replacement:\n"
    r"  .\.venv\Scripts\python.exe .\evaluate_oos.py"
)


def main() -> int:
    print(DEPRECATION_MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
