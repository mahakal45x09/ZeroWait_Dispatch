import matplotlib.pyplot as plt
import numpy as np

# 1. Setup the Scenarios
scenarios = ['Normal Operations', 'Hidden POS Rush\n(30+ Dine-in Tickets)', 'Heavy Rain\n(Waterlogging)']

# 2. Data: Actual time it took to cook the food
actual_prep_time = [20, 30, 35]

# 3. Data: Standard Zomato Model Predictions (Blind to POS & Weather)
standard_prediction = [20, 20, 20] 

# 4. Data: ZeroWait System Predictions (Enriched with POS & Weather APIs)
# (Base 20 + 10 POS Penalty) and (Base 20 + 15 Weather Penalty)
zerowait_prediction = [20, 30, 35] 

# 5. Calculate Rider Wait Times (Actual Prep Time - Predicted Prep Time)
standard_wait_time = [actual_prep_time[i] - standard_prediction[i] for i in range(len(scenarios))]
zerowait_wait_time = [actual_prep_time[i] - zerowait_prediction[i] for i in range(len(scenarios))]

# --- PLOTTING ---
x = np.arange(len(scenarios))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the Rider Wait Times
bars1 = ax.bar(x - width/2, standard_wait_time, width, label='Standard Zomato Model', color='#dc3545', edgecolor='black')
bars2 = ax.bar(x + width/2, zerowait_wait_time, width, label='ZeroWait AI Engine', color='#198754', edgecolor='black')

# Formatting the graph
ax.set_ylabel('Rider Wait Time at Restaurant (Minutes)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Real-World Chaos on Rider Wait Times\n(Standard Model vs. ZeroWait Engine)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=11)
ax.legend(fontsize=11)

# Add the exact minute labels on top of the bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} min',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(bars1)
autolabel(bars2)

# Add a subtle grid for premium look
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the image in high resolution
plt.tight_layout()
plt.savefig('zerowait_impact_graph.png', dpi=300)
plt.show()