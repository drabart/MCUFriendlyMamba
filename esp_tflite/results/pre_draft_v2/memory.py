import matplotlib.pyplot as plt
import numpy as np

# --- 1. SET UP GLOBAL DESIGN VARIABLES ---
colors = ['#2b5c8f', '#f03b20']  # Dark Blue (Base/Part 1), Warm Coral (Part 2)
font_label = {'fontsize': 11, 'family': 'sans-serif'}
font_title = {'fontsize': 12, 'family': 'sans-serif', 'fontweight': 'bold', 'loc': 'left'}

def style_axes(ax, x_label):
    """Applies a consistent clean journal style to the axes."""
    ax.set_xlabel(x_label, **font_label)
    ax.set_ylabel('Model Configuration', **font_label)
    ax.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

def plot_split_bar(ax, labels, part1_vals, part2_vals=None):
    """Generates the split horizontal bars and adds clean text annotations."""
    y_pos = np.arange(len(labels))
    
    if part2_vals is None:
        # For standard non-split models, draw a single uniform bar
        ax.barh(y_pos, part1_vals, height=0.5, color=colors[0])
        total_vals = part1_vals
        
        for i, val in enumerate(total_vals):
            ax.text(val + (max(total_vals) * 0.01), i, f"Total: {val}", 
                    va='center', ha='left', fontsize=9, color='#333333')
    else:
        # For split models, stack part 2 on top of part 1
        ax.barh(y_pos, part1_vals, height=0.5, color=colors[0], label='TFLite Execution')
        ax.barh(y_pos, part2_vals, left=part1_vals, height=0.5, color=colors[1], label='External buffers')
        total_vals = [p1 + p2 for p1, p2 in zip(part1_vals, part2_vals)]
        
        for i, (p1, p2, tot) in enumerate(zip(part1_vals, part2_vals, total_vals)):
            ax.text(tot + (max(total_vals) * 0.01), i, f"{p1} + {p2}\nTotal: {tot}", 
                    va='center', ha='left', fontsize=8.5, color='#333333')
        ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper right')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, max(total_vals) * 1.25)  # Make room for text labels


# --- GRAPH 3: Split KWS Float vs Int8 ---
# Sorted descending based on split kws float total (114936 + 52224 = 167160)
fig3, ax3 = plt.subplots(figsize=(8.5, 3.5))
g3_labels = ['split kws float', 'split kws int8']
g3_p1 = [114936, 47356]
g3_p2 = [52224, 13056]

plot_split_bar(ax3, g3_labels, g3_p1, g3_p2)
style_axes(ax3, 'Execution Metrics')
plt.tight_layout()
plt.savefig('chart_7_split_kws.png', dpi=300, bbox_inches='tight')
