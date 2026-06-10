import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP DATA AND LABELS ---
# Structured as: (Model Name, Part 1/Core, Part 2/Offload or 0 if single bar)
har_models = [
    ('har float', 74280, 0),
    ('split har float', 39736, 10240),
    ('har int8', 48300, 0),
    ('split har int8', 29596, 2560)
]

# Calculate total times for sorting
totals = [p1 + p2 for _, p1, p2 in har_models]

# Sort all lists concurrently descending based on total time
sorted_data = sorted(zip(har_models, totals), key=lambda x: x[1], reverse=False)
har_models_sorted, totals_sorted = zip(*sorted_data)

# Extract individual components back out for plotting
labels = [item[0] for item in har_models_sorted]
part1_vals = [item[1] for item in har_models_sorted]
part2_vals = [item[2] for item in har_models_sorted]

# --- 2. PLOT GENERATION ---
fig, ax = plt.subplots(figsize=(9, 4.5))
y_pos = np.arange(len(labels))

colors = ['#2b5c8f', '#f03b20']  # Dark Blue, Warm Coral

# Loop through and plot individually to handle mixed split/non-split types
for i in range(len(labels)):
    if part2_vals[i] == 0:
        # Standard model: single uniform bar
        ax.barh(y_pos[i], part1_vals[i], height=0.55, color=colors[0], 
                label='Standard / Core' if i == len(labels)-1 else "") # Unique label for legend
        
        # Text annotation
        ax.text(part1_vals[i] + (max(totals_sorted) * 0.01), i, f"Total: {part1_vals[i]}", 
                va='center', ha='left', fontsize=9, color='#333333')
    else:
        # Split model: stacked bar
        ax.barh(y_pos[i], part1_vals[i], height=0.55, color=colors[0],
                label='Standard / Core' if i == len(labels)-1 else "")
        ax.barh(y_pos[i], part2_vals[i], left=part1_vals[i], height=0.55, color=colors[1],
                label='Pipeline Offload' if i == len(labels)-2 else "") # Unique label for legend
        
        # Text annotation showing the math breakdown
        ax.text(totals_sorted[i] + (max(totals_sorted) * 0.01), i, 
                f"{part1_vals[i]} + {part2_vals[i]}\nTotal: {totals_sorted[i]}", 
                va='center', ha='left', fontsize=8.5, color='#333333')

# --- 3. AXES & STYLING ---
ax.set_title('(a) Comprehensive HAR Model Architecture Performance', 
             fontsize=12, family='sans-serif', fontweight='bold', loc='left')
ax.set_xlabel('Execution Metrics', fontsize=11, family='sans-serif')
ax.set_ylabel('Model Configuration', fontsize=11, family='sans-serif')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10, family='sans-serif')
ax.set_xlim(0, max(totals_sorted) * 1.25)  # Buffer zone for labels

# Clean layout adjustments for papers
ax.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Add clear legend explaining the colors
ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='lower right', fontsize=9.5)

plt.tight_layout()
plt.savefig('chart_8_all_har_combined.png', dpi=300, bbox_inches='tight')
