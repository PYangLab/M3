#!/usr/bin/env python3
"""Trim the per-epoch training logs in the rendered tutorial pages.

The engine prints one `Epoch N, Validation Loss: ...` line per epoch (500 by
default), and on the Python side each epoch is also accompanied by a tqdm
progress-bar frame on stderr. Left as-is, one training cell scrolls for
hundreds of lines. This post-processes the *rendered* pages so each training
log keeps the first 3 and last 3 epochs plus a `... (N epochs omitted) ...`
marker, and drops the tqdm progress noise from the notebooks.

Run from the website/ directory after (re-)rendering:

    python tools/trim_epoch_logs.py

Idempotent — safe to run repeatedly (an already-trimmed log has < 8 epoch
lines, so it is left untouched).
"""
import re
import json
import glob

KEEP_HEAD = 3
KEEP_TAIL = 3
MIN_RUN = KEEP_HEAD + KEEP_TAIL + 2  # only collapse runs longer than this

EPOCH_RE = re.compile(r'^(#> )?Epoch \d+, Validation Loss:')

R_GLOBS = ['docs/notebooks/r_*.md', '_stash_gse/r_*.md']
PY_GLOBS = ['docs/notebooks/py_*.ipynb', '_stash_gse/py_*.ipynb']


def _collapse_lines(lines):
    """Collapse runs of consecutive epoch lines (list of str, no trailing \\n)."""
    out, i, n = [], 0, len(lines)
    while i < n:
        if EPOCH_RE.match(lines[i]):
            j = i
            while j < n and EPOCH_RE.match(lines[j]):
                j += 1
            run = lines[i:j]
            if len(run) > MIN_RUN:
                pref = '#> ' if run[0].startswith('#> ') else ''
                out += run[:KEEP_HEAD]
                out.append(f'{pref}...  ({len(run) - KEEP_HEAD - KEEP_TAIL} epochs omitted)  ...')
                out += run[-KEEP_TAIL:]
            else:
                out += run
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


def trim_r_md():
    changed = 0
    for f in sum((glob.glob(g) for g in R_GLOBS), []):
        text = open(f, encoding='utf-8').read()
        new = '\n'.join(_collapse_lines(text.split('\n')))
        if new != text:
            open(f, 'w', encoding='utf-8').write(new)
            changed += 1
    return changed


def _text_of(o):
    tx = o.get('text')
    return ''.join(tx) if isinstance(tx, list) else (tx or '')


def _is_tqdm(o):
    return (o.get('output_type') == 'stream' and o.get('name') == 'stderr'
            and _text_of(o).startswith('\r'))


def _is_pure_epoch(o):
    if o.get('output_type') != 'stream' or o.get('name') != 'stdout':
        return False
    s = _text_of(o)
    return s.count('\n') <= 1 and bool(EPOCH_RE.match(s))


def trim_py_ipynb():
    changed = 0
    for f in sum((glob.glob(g) for g in PY_GLOBS), []):
        nb = json.load(open(f, encoding='utf-8'))
        touched = False
        for c in nb.get('cells', []):
            outs = c.get('outputs')
            if not outs:
                continue
            outs2 = [o for o in outs if not _is_tqdm(o)]  # drop tqdm progress frames
            new, i, n = [], 0, len(outs2)
            while i < n:
                if _is_pure_epoch(outs2[i]):
                    j = i
                    while j < n and _is_pure_epoch(outs2[j]):
                        j += 1
                    run = outs2[i:j]
                    if len(run) > MIN_RUN:
                        ell = dict(run[0])
                        ell['text'] = [f'...  ({len(run) - KEEP_HEAD - KEEP_TAIL} epochs omitted)  ...\n']
                        new += run[:KEEP_HEAD] + [ell] + run[-KEEP_TAIL:]
                    else:
                        new += run
                    i = j
                else:
                    new.append(outs2[i])
                    i += 1
            if new != outs:
                c['outputs'] = new
                touched = True
        if touched:
            json.dump(nb, open(f, 'w', encoding='utf-8'), indent=1)
            changed += 1
    return changed


if __name__ == '__main__':
    r = trim_r_md()
    p = trim_py_ipynb()
    print(f'trimmed epoch logs: {r} R page(s), {p} Python notebook(s)')
