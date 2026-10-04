"""The calculator's results as a structured text file, and a reader for it.

The format is specified in ``docs/report-format/README.md``; this module is
its reference implementation, and :func:`parse_report` is what another program
should copy (or import) to consume one.  In short::

    #PROPWASH-REPORT 1
    [SECTION]
    key = <value> [unit] {CODE}
    [TABLE.NAME]
    col_a[unit],col_b[unit]
    1.0,2.0
    [END_TABLE]
    [INTEGRITY]
    body_sha256 = "<hex>"
    [END]

``<value>`` is a JSON scalar: a number, a double-quoted string, ``true``,
``false`` or ``null``.  ``{CODE}`` appears only on ``null`` values and says why
the value is not available.  Lines starting with ``#`` are comments.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

FORMAT_NAME = "PROPWASH-REPORT"
FORMAT_VERSION = 1
MAGIC = f"#{FORMAT_NAME} {FORMAT_VERSION}"

_KEY = r"[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*"
_LINE = re.compile(
    r"^(?P<key>" + _KEY + r") = (?P<value>null|true|false|-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
    r'|"(?:[^"\\]|\\.)*")(?: \[(?P<unit>[^\]]*)\])?(?: \{(?P<code>[A-Z][A-Z0-9_]*)\})?$')
_SECTION = re.compile(r"^\[(?P<name>[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)?)\]$")
_HEADER_CELL = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)\[(?P<unit>[^\]]*)\]$")


@dataclass(slots=True)
class Item:
    value: Any                   # float | int | str | bool | None
    unit: str = ""
    code: str | None = None      # reason code when value is None


@dataclass(slots=True)
class Table:
    columns: list[tuple[str, str]]          # (name, unit)
    rows: list[list[Any]]
    comment: str = ""


@dataclass(slots=True)
class Notice:
    code: str
    severity: str                # "error" | "warning" | "info"
    message: str
    key: str = ""


@dataclass(slots=True)
class Report:
    """Ordered sections of key/value items, tables, and notices."""

    sections: dict[str, dict[str, Item]] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)
    notices: list[Notice] = field(default_factory=list)
    #: In-memory objects for front ends (geometry, solver result); never written.
    extras: dict[str, Any] = field(default_factory=dict)

    # -- building ------------------------------------------------------------
    def section(self, name: str) -> dict[str, Item]:
        return self.sections.setdefault(name, {})

    def put(self, section: str, key: str, value: Any, unit: str = "",
            code: str | None = None) -> None:
        """Add one value.  Non-finite numbers become null; null needs a code."""
        if isinstance(value, float) and not math.isfinite(value):
            value = None
            code = code or "NOT_COMPUTED"
        if value is None and code is None:
            code = "NOT_COMPUTED"
        if value is not None:
            code = None
        self.section(section)[key] = Item(_plain(value), unit, code)

    def notice(self, code: str, severity: str, message: str, key: str = "") -> None:
        if not any(n.code == code and n.key == key and n.message == message
                   for n in self.notices):
            self.notices.append(Notice(code, severity, message, key))

    @property
    def status(self) -> str:
        sev = {n.severity for n in self.notices}
        return "error" if "error" in sev else "warning" if "warning" in sev else "ok"

    def get(self, section: str, key: str) -> Any:
        return self.sections[section][key].value

    # -- export --------------------------------------------------------------
    def to_text(self) -> str:
        return write_report(self)

    def save(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(self.to_text(), encoding="utf-8", newline="\n")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view: the same structure :func:`parse_report` returns."""
        return parse_report(self.to_text())


def _plain(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:  # pragma: no cover
        pass
    return v


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    v = _plain(v)
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "null"
        if v == 0.0:
            return "0"
        return f"{v:.6g}" if 1e-4 <= abs(v) < 1e7 else f"{v:.6e}"
    return json.dumps(str(v), ensure_ascii=True)


def _line(key: str, item: Item) -> str:
    out = f"{key} = {_fmt(item.value)}"
    if item.unit:
        out += f" [{item.unit}]"
    if item.value is None and item.code:
        out += f" {{{item.code}}}"
    return out


def write_report(report: Report) -> str:
    lines = [MAGIC,
             "# Structured results; specification: docs/report-format/README.md",
             "# Lines starting with '#' are comments. Values are JSON scalars; [unit]; {CODE} explains a null."]
    meta = report.sections.get("META", {})
    meta["status"] = Item(report.status)
    meta["notice_count"] = Item(len(report.notices))
    for name, items in report.sections.items():
        lines.append("")
        lines.append(f"[{name}]")
        for key, item in items.items():
            lines.append(_line(key, item))
    for name, table in report.tables.items():
        lines.append("")
        lines.append(f"[TABLE.{name}]")
        if table.comment:
            for c in table.comment.splitlines():
                lines.append(f"# {c}")
        lines.append(",".join(f"{n}[{u}]" for n, u in table.columns))
        for row in table.rows:
            lines.append(",".join(_fmt(v) for v in row))
        lines.append("[END_TABLE]")
    lines.append("")
    lines.append("[NOTICES]")
    for i, n in enumerate(report.notices, 1):
        lines.append(f"n{i:03d}.code = {_fmt(n.code)}")
        lines.append(f"n{i:03d}.severity = {_fmt(n.severity)}")
        lines.append(f"n{i:03d}.field = {_fmt(n.key) if n.key else 'null {NONE}'}")
        lines.append(f"n{i:03d}.message = {_fmt(n.message)}")
    body = "\n".join(lines) + "\n"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return body + f"\n[INTEGRITY]\nbody_sha256 = \"{digest}\"\n[END]\n"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class ReportFormatError(ValueError):
    pass


def _scalar(text: str) -> Any:
    if text == "null":
        return None
    if text in ("true", "false"):
        return text == "true"
    if text.startswith('"'):
        return json.loads(text)
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return float(text)


def _cell(text: str) -> Any:
    text = text.strip()
    if text.startswith('"'):
        return json.loads(text)
    return _scalar(text)


def _split_row(row: str) -> list[str]:
    """Split a table row on commas outside double-quoted strings."""
    cells, buf, quoted, escaped = [], [], False, False
    for ch in row:
        if quoted:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
            buf.append(ch)
        elif ch == ",":
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf))
    return cells


def parse_report(text: str, verify: bool = True) -> dict[str, Any]:
    """Read a report into plain Python structures.

    Returns ``{"format": {...}, "sections": {name: {key: {"value", "unit",
    "code"}}}, "tables": {name: {"columns": [{"name", "unit"}], "rows":
    [[...]]}}, "notices": [{"code", "severity", "field", "message"}],
    "integrity": {"ok": bool, ...}}``.  With ``verify`` a checksum mismatch
    raises :class:`ReportFormatError` (a hand-edited file); pass
    ``verify=False`` to read it anyway.
    """
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    if not lines or not lines[0].startswith(f"#{FORMAT_NAME} "):
        raise ReportFormatError(f"not a {FORMAT_NAME} file (first line must be '{MAGIC}')")
    try:
        version = int(lines[0].split()[1].split(".")[0])
    except (IndexError, ValueError):
        raise ReportFormatError("unreadable format version") from None
    if version > FORMAT_VERSION:
        raise ReportFormatError(f"format version {version} is newer than this reader "
                                f"({FORMAT_VERSION})")

    sections: dict[str, dict[str, Any]] = {}
    tables: dict[str, dict[str, Any]] = {}
    current: str | None = None
    table: dict[str, Any] | None = None
    integrity_line = None
    ended = False

    for n, raw in enumerate(lines, 1):
        line = raw.rstrip()
        if ended:
            if line.strip():
                raise ReportFormatError(f"line {n}: content after [END]")
            continue
        if not line or line.startswith("#"):
            continue
        if table is not None:
            if line == "[END_TABLE]":
                table = None
                continue
            cells = _split_row(line)
            if table["columns"] is None:
                cols = []
                for c in cells:
                    m = _HEADER_CELL.match(c.strip())
                    if not m:
                        raise ReportFormatError(f"line {n}: bad column header {c!r}")
                    cols.append({"name": m["name"], "unit": m["unit"]})
                table["columns"] = cols
                continue
            if len(cells) != len(table["columns"]):
                raise ReportFormatError(f"line {n}: {len(cells)} cells, expected "
                                        f"{len(table['columns'])}")
            table["rows"].append([_cell(c) for c in cells])
            continue
        m = _SECTION.match(line)
        if m:
            name = m["name"]
            if name == "END":
                ended = True
                continue
            if name == "INTEGRITY":
                integrity_line = n
            if name.startswith("TABLE."):
                table = {"columns": None, "rows": []}
                tables[name[len("TABLE."):]] = table
                current = None
                continue
            current = name
            sections.setdefault(name, {})
            continue
        m = _LINE.match(line)
        if not m or current is None:
            raise ReportFormatError(f"line {n}: cannot parse {line!r}")
        sections[current][m["key"]] = {"value": _scalar(m["value"]), "unit": m["unit"] or "",
                                       "code": m["code"]}

    if not ended:
        raise ReportFormatError("missing [END] line (file truncated?)")
    if table is not None:
        raise ReportFormatError("table not closed with [END_TABLE]")

    integrity = {"ok": None, "expected": None, "actual": None}
    if integrity_line is not None and "INTEGRITY" in sections:
        body = "\n".join(lines[:integrity_line - 1])
        # the writer puts exactly one blank line between the body and [INTEGRITY]
        if body.endswith("\n"):
            body = body[:-1]
        body += "\n"
        actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
        expected = sections["INTEGRITY"].get("body_sha256", {}).get("value")
        integrity = {"ok": actual == expected, "expected": expected, "actual": actual}
        if verify and not integrity["ok"]:
            raise ReportFormatError("checksum mismatch: the file was edited or damaged "
                                    "(parse with verify=False to read it anyway)")

    notices = []
    raw_n = sections.pop("NOTICES", {})
    ids = sorted({k.split(".")[0] for k in raw_n})
    for i in ids:
        notices.append({f: raw_n.get(f"{i}.{f}", {}).get("value")
                        for f in ("code", "severity", "field", "message")})
    sections.pop("INTEGRITY", None)
    return {"format": {"name": FORMAT_NAME, "version": version}, "sections": sections,
            "tables": tables, "notices": notices, "integrity": integrity}


def values(parsed: dict[str, Any], section: str) -> dict[str, Any]:
    """Just the values of one section, ``{key: value}``."""
    return {k: v["value"] for k, v in parsed["sections"].get(section, {}).items()}


def table_records(parsed: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """A table as a list of ``{column: value}`` records."""
    t = parsed["tables"][name]
    names = [c["name"] for c in t["columns"]]
    return [dict(zip(names, row)) for row in t["rows"]]


__all__ = ["Report", "Item", "Table", "Notice", "write_report", "parse_report",
           "ReportFormatError", "values", "table_records", "FORMAT_NAME",
           "FORMAT_VERSION", "MAGIC"]
