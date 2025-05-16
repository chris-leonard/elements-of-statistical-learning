import seaborn as sns
import matplotlib.pyplot as plt

def set_default_plot_style():
    """
    Set the default seaborn plotting style for consistent visualizations.
    """
    sns.set_theme(
        style="whitegrid",
        rc={
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.5,  # Major gridline opacity
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.bottom': True,
            'axes.spines.left': True,
            'figure.figsize': (10, 6)  # Default figure size
        }
    )
