#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_res(path):
    text = Path(path).read_text()
    # Blocks separated by blank line, first line may be size label
    entries = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        # first line maybe a size label like '50k' or '100k'
        size_line = lines[0]
        m = re.match(r"([0-9\.]+)\s*(k|M)?", size_line, re.IGNORECASE)
        idx = 1
        if m:
            num = float(m.group(1))
            mult = m.group(2)
            if mult and mult.lower() == 'k':
                size = int(num * 1_000)
            elif mult and mult.lower() == 'm':
                size = int(num * 1_000_000)
            else:
                size = int(num)
        else:
            # try to find size in the first INFO line
            size = None

        # find lines with data_const and data_mut
        const_us = None
        mut_us = None
        for l in lines:
            m_const = re.search(r"data_const:.*elapsed:\s*([0-9]+)\s*us", l)
            if m_const:
                const_us = int(m_const.group(1))
            m_mut = re.search(r"data_mut:.*elapsed:\s*([0-9]+)\s*us", l)
            if m_mut:
                mut_us = int(m_mut.group(1))

        # fallback: if first line wasn't size, try to parse a standalone number line
        if size is None:
            for l in lines:
                if re.match(r"^\d+$", l):
                    size = int(l)
                    break

        if size is None:
            # skip if no size
            continue

        entries.append((size, const_us, mut_us))

    return entries

def plot(entries, out='res_plot.png'):
    entries = sorted(entries, key=lambda x: x[0])
    sizes = [e[0] for e in entries]
    consts = [e[1] for e in entries]
    muts = [e[2] for e in entries]

    fig, ax = plt.subplots(figsize=(7,4))
    # Map data_const -> Flash, data_mut -> DRAM
    ax.plot(sizes, consts, marker='o', label='Flash (us)')
    ax.plot(sizes, muts, marker='o', label='DRAM (us)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Array size (elements)')
    ax.set_ylabel('Elapsed time (microseconds)')
    ax.set_title('Flash vs DRAM speed')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out)
    print(f"Wrote {out}")

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'res.txt'
    entries = parse_res(path)
    if not entries:
        print('No entries parsed from', path)
        sys.exit(1)
    plot(entries)
