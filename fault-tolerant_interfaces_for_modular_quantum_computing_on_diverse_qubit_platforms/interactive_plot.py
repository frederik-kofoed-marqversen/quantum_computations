import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from bisect import bisect_left

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load in data
path = "data/plot_pd.dat"
absolute_label_locations = [(100, 7000), (0.6, 5000), (0.1, 10_000)]  # plot_12.dat
with open(SCRIPT_DIR + "/" + path, 'r') as file:
    data = json.load(file)

x = np.array(data["x"])
y = np.array(data["y"])
X, Y = np.meshgrid(x, y, indexing='ij')
rs = [np.array(Z) for Z in data["rates"]]
labels = data["labels"]

# Prepare data for main plot
Z = np.stack(rs)
ids = np.argmax(Z, axis=0)

# ng_region = np.argmax(Z[np.array([0, 1, 3]), ...], axis=0)
# ng_region = (ng_region == 2).astype(int)

Z = np.max(Z, axis=0)
ids[Z==0] = -1

# Prepare figure and axes
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 0.2])
main_ax = fig.add_subplot(gs[0, :]) # spans both columns
x_slice_ax = fig.add_subplot(gs[1, 0])
y_slice_ax = fig.add_subplot(gs[1, 1])
x_slider_ax = fig.add_subplot(gs[2, 0])
y_slider_ax = fig.add_subplot(gs[2, 1])

# Main 3D view
mesh1 = main_ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
c = main_ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', norm='log')
fig.colorbar(c, ax=main_ax, label='r_distributed/r_physical')
# Plot region borders
for id, lab, loc in zip(np.unique(ids), labels, absolute_label_locations):
    main_ax.contour(X, Y, ids==id, levels=[0.5], colors='black', linewidths=1, corner_mask=False, linestyles="-")
    if loc:
        plt.text(*loc, lab + " regime", color='black', fontsize=12, fontweight='bold', ha='left', va='center')
# ng_contour = main_ax.contour(X, Y, ng_region, levels=[0.5], colors='black', linewidths=1, corner_mask=False, linestyles="dashed")

# Add region labels
for id, (label, loc) in enumerate(zip(labels, absolute_label_locations)):
    main_ax.text(*loc, label + " regime", color='black', fontsize=12, fontweight='bold', ha='left', va='center')
# Styling
# main_ax.set_ylim(y[0], 13_000)
main_ax.set_xlabel('r_bell/r_physical')
main_ax.set_ylabel('Allocated memory for networking')
main_ax.set_xscale("log")

# Initial slice value
init_x_index = len(x) // 2
init_x_slice = x[init_x_index]
init_y_index = len(y) // 2
init_y_slice = y[init_y_index]

# Plot slice lines
x_slice_indicator = main_ax.axvline(init_x_slice, linestyle="--", color="k", linewidth=0.5)
y_slice_indicator = main_ax.axhline(init_y_slice, linestyle="--", color="k", linewidth=0.5)

# X-slice view
x_slice_lines = []
for Z, label in zip(rs, labels):
    line, = x_slice_ax.plot(y, Z[init_x_index, :], label=label)
    x_slice_lines.append(line)
x_slice_ax.set_xlim(y[0], y[-1])
x_slice_ax.set_ylim(2e-5, 2)
x_slice_ax.set_yscale("log")
x_slice_ax.set_xlabel("Allocated memory for networking")
x_slice_ax.set_ylabel("r_distributed / r_physical")
x_slice_ax.legend(loc="lower right", fontsize=8)

# Y-slice view
y_slice_lines = []
for Z, label in zip(rs, labels):
    line, = y_slice_ax.plot(x, Z[:, init_y_index], label=label)
    y_slice_lines.append(line)
y_slice_ax.set_xlim(x[0], x[-1])
y_slice_ax.set_ylim(2e-5, 2)
y_slice_ax.set_xscale("log")
y_slice_ax.set_yscale("log")
y_slice_ax.set_xlabel("r_bell / r_physical")
y_slice_ax.set_ylabel("r_distributed / r_physical")
y_slice_ax.legend(loc="lower right", fontsize=8)

# Sliders
x_slider = Slider(x_slider_ax, 'log(r_bell)', np.log10(x[0]), np.log10(x[-1]), valinit=np.log10(init_x_slice), valstep=0.01)
y_slider = Slider(y_slider_ax, 'memory', y[0], y[-1], valinit=init_y_slice, valstep=1)

plt.tight_layout()

# Update functions
def x_update(val):
    # Update x slice
    x_slice = 10**x_slider.val
    idx = bisect_left(x, x_slice)
    x_slice_indicator.set_xdata([x[idx], x[idx]])
    for line, Z in zip(x_slice_lines, rs):
        line.set_ydata(Z[idx, :])
    fig.canvas.draw_idle()

def y_update(val):
    # Update y slice
    y_slice = y_slider.val
    idx = bisect_left(y, y_slice)
    y_slice_indicator.set_ydata([y[idx], y[idx]])
    for line, Z in zip(y_slice_lines, rs):
        line.set_ydata(Z[:, idx])
    fig.canvas.draw_idle()

x_slider.on_changed(x_update)
y_slider.on_changed(y_update)
plt.show()
