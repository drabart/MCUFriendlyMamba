from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Read and parse the file
file_path = Path(__file__).with_name("group_compare.txt")
output_path = Path(__file__).with_name("group_compare.png")
with file_path.open("r") as f:
    raw_groups = [block for block in f.read().strip().split("\n\n") if block.strip()]


def split_groups(raw_groups):
    if not raw_groups:
        raise ValueError(f"No data found in {file_path}")

    if len(raw_groups) != 2:
        raise ValueError(f"Expected 2 groups in {file_path}, found {len(raw_groups)}")

    return [block.splitlines() for block in raw_groups]


group1, group2 = split_groups(raw_groups)   # int8 (top), float (bottom)

def parse_group(group):
    times_by_op = {}
    order = []
    for line in group:
        parts = line.split(" took ")
        if len(parts) != 2:
            raise ValueError(f"Unrecognized line format: {line}")
        op = parts[0]
        time = int(parts[1].replace(" us.", ""))
        order.append(op)
        times_by_op[op] = time
    return order, times_by_op

ops1, times1 = parse_group(group1)
ops2, times2 = parse_group(group2)

ops = sorted(ops1, key=lambda op: times1[op], reverse=True)
int8_times = [times1[op] for op in ops]
float_times = [times2.get(op, 0) for op in ops]
missing_float_ops = [op for op in ops if op not in times2]
x = np.arange(len(ops))
width = 0.35

if plt is None:
    print('matplotlib is not installed; showing aligned values instead of plotting:\n')
    for op, int8_time, float_time in zip(ops, int8_times, float_times):
        if op in times2:
            float_display = f'{int(float_time)} us'
        else:
            float_display = 'empty'
        print(f'{op}: int8={int8_time} us, float={float_display}')
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, int8_times, width, label='int8')
    rects2 = ax.bar(x + width/2, float_times, width, label='float')

    ax.set_ylabel('Time (us)')
    ax.set_title('Operation Time Comparison (int8 vs float)')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved graph to {output_path}')