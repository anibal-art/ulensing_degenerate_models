#!/usr/bin/env python3
"""Build a module from top-level code of an existing file, verbatim.

extract_module.py SRC OUT --header H [--middle M] --functions f1 f2 ...
                  [--globals g1 g2 ...]

OUT = header file + functions (source order, verbatim) + middle file +
top-level statements assigning --globals (source order, verbatim).
"""
import argparse, ast

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--header", required=True)
ap.add_argument("--middle", default=None)
ap.add_argument("--functions", nargs="*", default=[])
ap.add_argument("--globals", nargs="*", default=[])
a = ap.parse_args()

lines = open(a.src, encoding="utf-8").read().splitlines(keepends=True)
tree = ast.parse("".join(lines))

def segment(node):
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(lines[start - 1:node.end_lineno])

def assigned(node):
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()

want_f, want_g = list(a.functions), set(a.globals)
found_f = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in want_f}
missing = [f for f in want_f if f not in found_f]
if missing:
    raise SystemExit(f"functions not found: {missing}")
g_nodes = [n for n in tree.body if assigned(n) & want_g]
missing = want_g - set().union(*[assigned(n) for n in g_nodes]) if g_nodes else want_g
if missing:
    raise SystemExit(f"globals not found: {sorted(missing)}")

parts = [open(a.header, encoding="utf-8").read().rstrip("\n") + "\n"]
for n in sorted(found_f.values(), key=lambda n: n.lineno):
    parts.append("\n\n" + segment(n))
if a.middle:
    parts.append("\n\n" + open(a.middle, encoding="utf-8").read().rstrip("\n") + "\n")
for n in g_nodes:
    parts.append("\n" + segment(n))
open(a.out, "w", encoding="utf-8").write("".join(parts).rstrip("\n") + "\n")
print(a.out, "functions:", len(found_f), "global statements:", len(g_nodes))
