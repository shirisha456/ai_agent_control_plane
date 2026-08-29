"""The domain layer must stay free of I/O.

Not pedantry: the moment a database import creeps into acp.domain, the
transition table stops being testable in milliseconds and the "policy is pure,
mechanism is SQL" split -- the thing that lets the scheduler be unit-tested
without a cluster -- quietly dies.
"""

from __future__ import annotations

import ast
import pathlib

DOMAIN = pathlib.Path(__file__).resolve().parents[2] / "src" / "acp" / "domain"
FORBIDDEN_PREFIXES = ("acp.db", "acp.api", "sqlalchemy", "psycopg", "fastapi", "httpx", "socket")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_domain_has_no_io_imports() -> None:
    offenders: list[str] = []
    for path in DOMAIN.rglob("*.py"):
        for mod in _imported_modules(path):
            if mod.startswith(FORBIDDEN_PREFIXES):
                offenders.append(f"{path.name} imports {mod}")
    assert not offenders, "domain layer must stay pure: " + "; ".join(offenders)
