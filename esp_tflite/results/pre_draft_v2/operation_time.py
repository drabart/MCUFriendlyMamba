import matplotlib.pyplot as plt
import numpy as np

# Re-establishing the dataframe from previous step
import pandas as pd
df = pd.DataFrame([
    {"Operation": "SUM", "Float": 18270, "Int8_ANSI": 18151, "Int8": 18152},
    {"Operation": "EXP", "Float": 15157, "Int8_ANSI": 12142, "Int8": 12230},
    {"Operation": "FULLY_CONNECTED", "Float": 15090, "Int8_ANSI": 25595, "Int8": 6657},
    {"Operation": "MUL", "Float": 10890, "Int8_ANSI": 59897, "Int8": 56227},
    {"Operation": "DEPTHWISE_CONV_2D", "Float": 4657, "Int8_ANSI": 2646, "Int8": 716},
    {"Operation": "ADD", "Float": 1710, "Int8_ANSI": 20875, "Int8": 9530},
    {"Operation": "SLICE", "Float": 1633, "Int8_ANSI": 2913, "Int8": 2931},
    {"Operation": "RESHAPE", "Float": 394, "Int8_ANSI": 250, "Int8": 258},
    {"Operation": "TRANSPOSE", "Float": 315, "Int8_ANSI": 294, "Int8": 296},
    {"Operation": "PAD", "Float": 255, "Int8_ANSI": 272, "Int8": 269},
    {"Operation": "CONCATENATION", "Float": 72, "Int8_ANSI": 72, "Int8": 69},
    {"Operation": "RELU", "Float": 68, "Int8_ANSI": 671, "Int8": 671},
    {"Operation": "GATHER_ND", "Float": 51, "Int8_ANSI": 48, "Int8": 48},
    {"Operation": "QUANTIZE", "Float": 0, "Int8_ANSI": 12717, "Int8": 12729},
    {"Operation": "DEQUANTIZE", "Float": 0, "Int8_ANSI": 14345, "Int8": 14400}
])

# For horizontal bar charts, to have descending order from top to bottom, 
# we invert the dataframe or invert the axis. Let's invert the dataframe so the highest is first.
df_sorted = df.iloc[::-1].reset_index(drop=True)

y = np.arange(len(df_sorted))
width = 0.25

# Let's create a beautiful linear plot with value labels where appropriate, or just clean bars.
# Since the user asked for a paper figure, let's refine the linear one to look extremely polished.

fig, ax = plt.subplots(figsize=(10, 8))

# Define clean professional colors
colors = ['#2b5c8f', '#74a9cf', '#f03b20'] # Dark blue, light blue, warm red/orange

rects1 = ax.barh(y + width, df_sorted['Float'], width, label='Float', color=colors[0])
rects2 = ax.barh(y, df_sorted['Int8_ANSI'], width, label='Int8 ANSI', color=colors[1])
rects3 = ax.barh(y - width, df_sorted['Int8'], width, label='Int8', color=colors[2])

ax.set_xlabel('Execution Time ($\mu$s)', fontsize=12, family='sans-serif')
ax.set_ylabel('Operation', fontsize=12, family='sans-serif')
ax.set_yticks(y)
ax.set_yticklabels(df_sorted['Operation'], fontsize=10, family='sans-serif')

# Clean up the spines (box)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.grid(axis='x', linestyle=':', alpha=0.6, color='gray')
ax.set_axisbelow(True)

# Place legend in a clean spot
ax.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.savefig('operation_performance_linear.png', dpi=300, bbox_inches='tight')
print("Linear plot updated.")