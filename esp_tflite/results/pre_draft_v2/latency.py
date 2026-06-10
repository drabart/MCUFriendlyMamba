import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP GLOBAL DESIGN VARIABLES (Academic Style) ---
colors = ['#2b5c8f', '#f03b20']  # Dark Blue (TFLite), Warm Coral (Setup/Overhead)
font_label = {'fontsize': 11, 'family': 'sans-serif'}
font_title = {'fontsize': 12, 'family': 'sans-serif', 'fontweight': 'bold', 'loc': 'left'}

def style_axes(ax, x_label):
    """Applies a consistent clean journal style to an active subplot."""
    ax.set_xlabel(x_label, **font_label)
    ax.set_ylabel('Model', **font_label)
    ax.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='lower right')

def plot_stacked_bar(ax, labels, tflite_vals, total_vals):
    """Helper to generate the split stacked bars and annotate exact values."""
    y_pos = np.arange(len(labels))
    overhead_vals = [tot - tfl for tot, tfl in zip(total_vals, tflite_vals)]
    
    # Base TFLite Execution Bar
    ax.barh(y_pos, tflite_vals, height=0.5, color=colors[0], label='Inside TFLite')
    # Stacked Overhead Bar
    ax.barh(y_pos, overhead_vals, left=tflite_vals, height=0.5, color=colors[1], label='Setup & Overhead')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    
    # Add precise numerical labels to the end of each bar for papers
    for i, (tfl, tot) in enumerate(zip(tflite_vals, total_vals)):
        ax.text(tot + (max(total_vals) * 0.01), i, f"TFLite: {tfl}\nTotal: {tot}", 
                va='center', ha='left', fontsize=8.5, color='#333333')
        
    # Extend x-axis slightly so labels don't clip
    ax.set_xlim(0, max(total_vals) * 1.2)

# --- 2. INITIALIZE PLOT SUBGRIDS (4 Separate Figures) ---

# --- GRAPH 1: Floating Point Optimization Comparison ---
fig1, ax1 = plt.subplots(figsize=(8.5, 4))
g1_labels = ['har float opt', 'har float', 'har float noopt']
g1_tflite = [54921, 68562, 86287]
g1_total  = [55539, 69199, 86995]

plot_stacked_bar(ax1, g1_labels, g1_tflite, g1_total)
style_axes(ax1, 'Execution Time ($\mu$s)')
plt.tight_layout()
plt.savefig('chart_1_har_float_opt.png', dpi=300, bbox_inches='tight')


# --- GRAPH 2: HAR Datatype Performance Comparison ---
fig2, ax2 = plt.subplots(figsize=(8.5, 4))
g2_labels = ['har float', 'har int8', 'har int8 ansi']
g2_tflite = [68562, 135183, 170888]
g2_total  = [69199, 135925, 171627]

# Sorted descending based on har float baseline
plot_stacked_bar(ax2, g2_labels, g2_tflite, g2_total)
style_axes(ax2, 'Execution Time ($\mu$s)')
plt.tight_layout()
plt.savefig('chart_2_har_datatypes.png', dpi=300, bbox_inches='tight')


# --- GRAPH 3: HAR vs. SPLIT Architectural Comparison ---
fig3, ax3 = plt.subplots(figsize=(8.5, 4.5))
g3_labels = ['har float', 'split har float', 'har int8', 'split har int8']
g3_tflite = [68562, 73524, 135183, 144946]
g3_total  = [69199, 347103, 135925, 487058]

plot_stacked_bar(ax3, g3_labels, g3_tflite, g3_total)
style_axes(ax3, 'Execution Time ($\mu$s)')
plt.tight_layout()
plt.savefig('chart_3_har_vs_split.png', dpi=300, bbox_inches='tight')


# --- GRAPH 4: KeyWord Spotting (KWS) SPLIT Models ---
fig4, ax4 = plt.subplots(figsize=(8.5, 3.5))
g4_labels = ['split kws float', 'split kws int8']
g4_tflite = [316765, 678276]
g4_total  = [590740, 1074347]

plot_stacked_bar(ax4, g4_labels, g4_tflite, g4_total)
style_axes(ax4, 'Execution Time ($\mu$s)')
plt.tight_layout()
plt.savefig('chart_4_kws_split.png', dpi=300, bbox_inches='tight')

