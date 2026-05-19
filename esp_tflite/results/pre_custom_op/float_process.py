from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Read and parse the file
file_path = Path(__file__).with_name("group_compare.txt")
output_path = Path(__file__).with_name("float_compare.png")
with file_path.open("r") as f:
    raw_groups = [block for block in f.read().strip().split("\n\n") if block.strip()]


def split_groups(raw_groups):
    if not raw_groups:
        raise ValueError(f"No data found in {file_path}")

    if len(raw_groups) != 2:
        raise ValueError(f"Expected 2 groups in {file_path}, found {len(raw_groups)}")

    return [block.splitlines() for block in raw_groups]


_, float_group = split_groups(raw_groups)  # Extract only the float group


def parse_group(group):
    times_by_op = {}
    for line in group:
        parts = line.split(" took ")
        if len(parts) != 2:
            raise ValueError(f"Unrecognized line format: {line}")
        op = parts[0]
        time = int(parts[1].replace(" us.", ""))
        times_by_op[op] = time
    return times_by_op


times = parse_group(float_group)

# Sort by time descending
ops = sorted(times.keys(), key=lambda op: times[op], reverse=True)
float_times = [times[op] for op in ops]
x = np.arange(len(ops))

if plt is None:
    print('matplotlib is not installed; showing values sorted by time (descending):\n')
    for op, time in zip(ops, float_times):
        print(f'{op}: {time} us')
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, float_times, color='steelblue')

    ax.set_ylabel('Time (us)')
    ax.set_title('Operation Time (float) - Sorted by Time')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved graph to {output_path}')
