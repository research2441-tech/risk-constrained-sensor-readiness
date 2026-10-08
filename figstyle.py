import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.0, 'savefig.dpi': 400,
    'axes.grid': False, 'legend.frameon': False, 'pdf.fonttype': 42,
})
GRAYS = ['0.0', '0.35', '0.55', '0.72', '0.85']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
LINES = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
