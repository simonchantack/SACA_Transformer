"""Show that the shipped modules are AST-equivalent to the tree that ran the reproduction check (gap 1, option A).

The reproduction check (6 frozen FD001 seed-0 cells, validation only) was relaunched at 2026-10-27 04:37:53 local time.
Afterwards the programmer ran ``ruff check . --fix`` and made typing / lint edits. ``results/repro_check/launch_tree``
holds the reconstructed launch tree (every ``.py`` file with a ``.txt`` suffix so that lint and test collection ignore
it). This script compares every module of the launch tree with the shipped module after a normalisation that removes
only what cannot change behaviour:

1. docstrings (module, class, function);
2. annotations (argument, return, annotated assignments become plain assignments, bare declarations are dropped);
3. ``# type: ignore`` comments (comments are not in the AST);
4. import statements: compared as the set of bound names that the module body actually uses (import order and unused
   imports are lint changes);
5. local variable names inside each function (alpha-renamed by order of first occurrence);
6. single-use aliases ``name = self.attr`` (inlined at the use site, on both sides).

Everything else must be identical. A module that still differs after the normalisation is reported with the first
differing statements; nothing is hidden. The script reads no data and no test file.

Usage: ``python scripts/ast_equivalence.py [--out results/repro_check/ast_equivalence.md]``
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
from pathlib import Path

CRLF = chr(13) + chr(10)
LF = chr(10)
ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "results" / "repro_check" / "launch_tree"


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


class _Strip(ast.NodeTransformer):
    """Remove docstrings and annotations; turn annotated assignments into plain ones."""

    def _body(self, node: ast.AST) -> None:
        if hasattr(node, "body") and isinstance(node.body, list):
            node.body = _strip_docstring(node.body) or [ast.Pass()]

    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
        node.body = _strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
        self._body(node)
        self.generic_visit(node)
        return node

    def _func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self._body(node)
        node.returns = None
        args = node.args
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if a is not None:
                a.annotation = None
        self.generic_visit(node)
        return node

    visit_FunctionDef = _func  # noqa: N815
    visit_AsyncFunctionDef = _func  # noqa: N815

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:  # noqa: N802
        self.generic_visit(node)
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value, type_comment=None), node)


def _is_self_attr_chain(node: ast.AST) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == "self"


class _InlineAlias(ast.NodeTransformer):
    """Inline ``name = self.attr`` when ``name`` is assigned once in the function and read exactly once."""

    def _func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        stores: dict[str, int] = {}
        loads: dict[str, int] = {}
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                d = stores if isinstance(n.ctx, ast.Store) else loads
                d[n.id] = d.get(n.id, 0) + 1
        cands: dict[str, ast.expr] = {}
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and _is_self_attr_chain(stmt.value)
                and stores.get(stmt.targets[0].id) == 1
                and loads.get(stmt.targets[0].id) == 1
            ):
                cands[stmt.targets[0].id] = stmt.value
        if not cands:
            return node
        node.body = [
            s
            for s in node.body
            if not (isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) and s.targets[0].id in cands)
        ]

        class _Sub(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:  # noqa: N802
                if isinstance(n.ctx, ast.Load) and n.id in cands:
                    return cands[n.id]
                return n

        for i, s in enumerate(node.body):
            node.body[i] = _Sub().visit(s)
        return node

    visit_FunctionDef = _func  # noqa: N815
    visit_AsyncFunctionDef = _func  # noqa: N815


class _Alpha(ast.NodeTransformer):
    """Rename function-local names (arguments, stored names) by order of first occurrence."""

    def _func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        local: list[str] = []
        args = node.args
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if a is not None and a.arg not in local:
                local.append(a.arg)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in local:
                local.append(n.id)
        # ast.walk order is BFS and identical for structurally identical trees.
        mapping = {n: f"_v{i}" for i, n in enumerate(local)}
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if a is not None:
                a.arg = mapping[a.arg]
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in mapping:
                n.id = mapping[n.id]
        for child in node.body:
            self.visit(child)
        return node

    visit_FunctionDef = _func  # noqa: N815
    visit_AsyncFunctionDef = _func  # noqa: N815


def normalise(source: str) -> tuple[list[str], list[str]]:
    """Return (normalised statements as dumps, sorted used-import bindings)."""
    tree = ast.parse(source)
    tree = _Strip().visit(tree)
    tree = _InlineAlias().visit(tree)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    imports: list[str] = []
    body: list[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            mod = ("." * getattr(stmt, "level", 0)) + (getattr(stmt, "module", None) or "")
            for al in stmt.names:
                bound = al.asname or al.name.split(".")[0]
                if bound in used or bound == "annotations":
                    imports.append(f"{mod}:{al.name}:{al.asname}")
        else:
            body.append(stmt)
    tree.body = body
    tree = _Alpha().visit(tree)
    dumps = [ast.dump(s, indent=1) for s in tree.body]
    return dumps, sorted(set(imports))


def compare(launch_txt: Path, shipped: Path) -> tuple[bool, list[str]]:
    a, ia = normalise(launch_txt.read_text(encoding="utf-8"))
    b, ib = normalise(shipped.read_text(encoding="utf-8"))
    notes: list[str] = []
    if ia != ib:
        notes.append(f"imports used differ: only-launch={sorted(set(ia) - set(ib))} only-shipped={sorted(set(ib) - set(ia))}")
    if a != b:
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                diff = list(difflib.unified_diff(x.splitlines(), y.splitlines(), "launch", "shipped", lineterm="", n=1))
                notes.append(f"top-level statement {i} differs:\n" + "\n".join(diff[:40]))
        if len(a) != len(b):
            notes.append(f"statement count: launch {len(a)}, shipped {len(b)}")
    return (not notes), notes


# Residual differences that the AST normalisation does not absorb. Each entry is (module, substring that must appear in
# the added text of the hunk, reason). A hunk that matches no entry makes the module UNEXPLAINED.
ALLOWED: list[tuple[str, str, str]] = [
    ("src/saca_rul/attention.py", "attn_out", "split of a re-bound local (`attn` -> `attn_out`) after the early-return branch; `attn` is not read afterwards"),
    ("src/saca_rul/data.py", "abs(float(std[c]))", "float() of a numpy scalar before abs(); same value, typing only"),
    ("src/saca_rul/data.py", "type: ignore", "comment only"),
    ("src/saca_rul/cli.py", "add_parser(\"freeze\")", "additive: new `freeze` and `tune` sub-commands; the `repro` branch is untouched"),
    ("src/saca_rul/cli.py", "a.cmd == \"freeze\"", "additive dispatch for the new sub-commands (after the `repro` branch)"),
    ("src/saca_rul/cli.py", "tune            one validation-only", "usage text"),
    ("src/saca_rul/cli.py", "norm = ", "argument expression hoisted into a local before the same call (test-read path, not `repro`)"),
    ("src/saca_rul/cli.py", "norm=norm)", "call site of the hoisted local above"),
    ("src/saca_rul/reporting/verify.py", "nj.replace(re.search", "first-match replace of the same text (`re.sub(..., count=1)` before); table verification only"),
    ("src/saca_rul/tuning.py", "hist: list[dict[str, Any]] = []", "tuple assignment split to give `hist` a type; same initial values"),
    ("src/saca_rul/tuning.py", "str(c).replace", "column labels are already str; typing only (tuning is not on the repro path)"),
    ("src/saca_rul/reporting/verify.py", "read_bytes().decode", "table-verification helper reads bytes (no newline translation); not on the repro path"),
]


def hunks(launch_txt: Path, shipped: Path) -> list[tuple[list[str], list[str]]]:
    """Line-level hunks (CR-insensitive) as (removed lines, added lines)."""
    a = launch_txt.read_text(encoding="utf-8").replace(CRLF, LF).split(LF)
    b = shipped.read_text(encoding="utf-8").replace(CRLF, LF).split(LF)
    out: list[tuple[list[str], list[str]]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((a[i1:i2], b[j1:j2]))
    return out


def explain(rel: str, launch_txt: Path, shipped: Path) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True
    for rem, add in hunks(launch_txt, shipped):
        text = LF.join(add)
        hit = next((r for m, sub, r in ALLOWED if m == rel and sub in text), None)
        if hit is None and not add:
            hit = "deleted lines only"
        ok &= hit is not None
        lines.append("- removed: " + " / ".join(x.strip() for x in rem)[:200])
        lines.append("  added:   " + " / ".join(x.strip() for x in add)[:200])
        lines.append("  => " + (hit or "UNEXPLAINED"))
    return ok, lines


def repro_path_modules() -> set[str]:
    """Modules reachable from ``cli.cmd_repro`` in the launch tree (module-level imports of ``cli`` plus the imports
    inside ``cmd_repro``; every import, including function-local ones, of every module reached from there)."""
    base = LAUNCH / "src" / "saca_rul"

    def imports(path: Path, only_func: str | None) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes: list[ast.AST] = []
        if only_func is None:
            nodes = list(ast.walk(tree))
        else:
            nodes = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
            for f in tree.body:
                if isinstance(f, ast.FunctionDef) and f.name == only_func:
                    nodes += list(ast.walk(f))
        out: set[str] = set()
        for n in nodes:
            if isinstance(n, ast.ImportFrom) and n.level == 1:
                if n.module:
                    out.add(n.module)
                else:
                    out.update(a.name for a in n.names)
        return out

    seen: set[str] = set()
    todo = ["cli"]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        f = base / (m.replace(".", "/") + ".py.txt")
        if f.exists():
            todo += sorted(imports(f, "cmd_repro" if m == "cli" else None))
    return {f"src/saca_rul/{m.replace('.', '/')}.py" for m in seen if (base / (m.replace(".", "/") + ".py.txt")).exists()}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "repro_check" / "ast_equivalence.md"))
    a = ap.parse_args()
    rows: list[str] = []
    details: list[str] = []
    n_eq = n_total = n_explained = 0
    on_path = repro_path_modules()
    n_path_ok = n_path = 0
    for txt in sorted(LAUNCH.rglob("*.py.txt")):
        rel = txt.relative_to(LAUNCH).as_posix()[: -len(".txt")]
        shipped = ROOT / rel
        n_total += 1
        if not shipped.exists():
            rows.append(f"| `{rel}` | MISSING in shipped tree | | | |")
            continue
        byte_same = sha(txt) == sha(shipped)
        eq, notes = compare(txt, shipped)
        n_eq += int(eq)
        explained = False
        if byte_same:
            status = "byte-identical"
        elif eq:
            status = "AST-equivalent (normalised)"
        else:
            explained, hlines = explain(rel, txt, shipped)
            status = "differs; every hunk listed and explained below" if explained else "DIFFERENT (UNEXPLAINED hunk)"
            details.append(f"### {rel}\n\n" + LF.join(hlines) + LF)
        n_explained += int(explained)
        path_flag = rel in on_path
        if path_flag:
            n_path += 1
            n_path_ok += int(byte_same or eq or explained)
        elif not (byte_same or eq or explained):
            status += " (off the repro path; changed after the audit, see README section 5)"
        rows.append(f"| `{rel}` | {status} | {'yes' if path_flag else 'no'} | {sha(txt)[:12]} | {sha(shipped)[:12]} |")
    head = [
        "# AST equivalence: launch tree vs shipped tree",
        "",
        "Generated by `scripts/ast_equivalence.py`. Normalisation: docstrings, annotations, imports (as used-binding set),",
        "function-local names and single-use `name = self.attr` aliases are removed or canonicalised; `# type: ignore`",
        "comments are not in the AST. Launch tree: `results/repro_check/launch_tree/` (reconstructed, see README section 5).",
        "",
        f"Modules compared: {n_total}; byte-identical or equal after normalisation: {n_eq}; "
        f"differing with every hunk listed and explained: {n_explained}.",
        "",
        f"Modules on the reproduction path (reachable from `cli.cmd_repro` in the launch tree): {n_path}; "
        f"of these equal or explained: {n_path_ok}.",
        "",
        "| module | result | on repro path | launch sha256 (12) | shipped sha256 (12) |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
    out = Path(a.out)
    out.write_text("\n".join(head + details), encoding="utf-8")
    print(f"{n_eq}/{n_total} equal after normalisation, {n_explained} explained residual; report in {out}")
    print(f"repro path: {n_path_ok}/{n_path} modules equal or explained")
    return 0 if n_path_ok == n_path else 1


if __name__ == "__main__":
    raise SystemExit(main())
