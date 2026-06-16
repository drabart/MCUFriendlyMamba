import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP DATA AND LABELS ---
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

# --- USER DEFINED NEW FONT VARIABLES ---
font_label = {'fontsize': 19, 'family': 'sans-serif'}
FONT_TICK_SIZE = 14
FONT_ANNOTATION_SIZE = 15

# --- 2. PLOT GENERATION ---
# Width and height slightly increased to seamlessly hold the larger user font sizes
fig, ax = plt.subplots(figsize=(9.5, 5.0))
y_pos = np.arange(len(labels))

colors = ['#2b5c8f', '#f03b20']  # Dark Blue, Warm Coral

# Loop through and plot individually to handle mixed split/non-split types
for i in range(len(labels)):
    if part2_vals[i] == 0:
        # Standard model: single uniform bar
        ax.barh(y_pos[i], part1_vals[i], height=0.55, color=colors[0], 
                label='TFLite Execution' if i == len(labels)-1 else "") 
        
        # Text annotation using FONT_ANNOTATION_SIZE
        ax.text(part1_vals[i] + (max(totals_sorted) * 0.01), i, f"Total: {part1_vals[i]}", 
                va='center', ha='left', fontsize=FONT_ANNOTATION_SIZE, color='#333333')
    else:
        # Split model: stacked bar
        ax.barh(y_pos[i], part1_vals[i], height=0.55, color=colors[0],
                label='TFLite Execution' if i == len(labels)-1 else "")
        ax.barh(y_pos[i], part2_vals[i], left=part1_vals[i], height=0.55, color=colors[1],
                label='External buffers' if i == len(labels)-2 else "") 
        
        # Text annotation using FONT_ANNOTATION_SIZE
        ax.text(totals_sorted[i] + (max(totals_sorted) * 0.01), i, 
                f"{part1_vals[i]} + {part2_vals[i]}\nTotal: {totals_sorted[i]}", 
                va='center', ha='left', fontsize=FONT_ANNOTATION_SIZE, color='#333333')

# --- 3. AXES & STYLING ---
# Applied custom font_label dict with built-in padding to avoid tick overlaps
ax.set_xlabel('Execution Metrics', **font_label, labelpad=10)
ax.set_ylabel('Model Configuration', **font_label, labelpad=10)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=FONT_TICK_SIZE, family='sans-serif')

# Adjust tick label parameters cleanly
ax.tick_params(axis='both', which='major', labelsize=FONT_TICK_SIZE)

# Extend buffer zone to 1.30 to accommodate larger text values nicely
ax.set_xlim(0, max(totals_sorted) * 1.30)  

# Clean layout adjustments for papers
ax.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Clean, low-profile legend placed horizontally underneath the main plot
ax.legend(
    frameon=True, 
    facecolor='white', 
    edgecolor='none', 
    loc='upper center',          
    bbox_to_anchor=(0.5, -0.22), # Placed underneath with small vspacing 
    ncol=2,                      # Side-by-side arrangement
    fontsize=FONT_TICK_SIZE
)

plt.savefig('chart_8_all_har_combined.png', dpi=300, bbox_inches='tight')