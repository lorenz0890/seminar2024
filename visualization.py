import numpy as np
from matplotlib import pyplot as plt


def clean_nans(data_list):
    """Remove NaN values from a list."""
    return [x for x in data_list if not np.isnan(x)]

'''
def plot_similarity(results, filename='similarity_plot.png'):
    epochs = range(1, len(next(iter(results.values()))['intra']) + 1)  # Number of epochs

    # Create a single plot for intra and inter similarities
    plt.figure(figsize=(10, 6))

    # Plot intra-cluster similarity with solid lines
    for depth, metrics in results.items():
        intra_cleaned = clean_nans(metrics['intra'])
        plt.plot(range(1, len(intra_cleaned) + 1), intra_cleaned, label=f'Intra Depth {depth}', linestyle='-')

    # Plot inter-cluster similarity with dashed lines
    for depth, metrics in results.items():
        inter_cleaned = clean_nans(metrics['inter'])
        plt.plot(range(1, len(inter_cleaned) + 1), inter_cleaned, label=f'Inter Depth {depth}', linestyle='--')

    # Add title and labels
    plt.title('Intra and Inter Cluster Similarity over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity')

    # Show legend
    plt.legend()

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
'''


def plot_similarity(results, filename='similarity_with_accuracy_plot.png'):
    # Determine number of epochs based on intra/inter list length
    similarity_epochs = range(1, len(next(iter(results.values()))['intra']) + 1)  # Number of epochs for intra/inter

    # Determine number of epochs for test accuracy
    test_acc_epochs = range(1, len(results['test_acc']) + 1)  # Number of epochs for test_acc

    # Create a figure for intra/inter similarities and test accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Colors for different depths
    colors = ['b', 'g', 'r']  # You can add more colors if needed for more depths

    # Store lines for custom legend
    intra_lines = []
    inter_lines = []

    # Plot intra-cluster similarity with solid lines and inter-cluster similarity with dashed lines
    for i, (depth, metrics) in enumerate(results.items()):
        if isinstance(depth, int):  # Only process depths, not 'test_acc'
            intra_cleaned = clean_nans(metrics['intra'])
            inter_cleaned = clean_nans(metrics['inter'])

            # Plot intra with solid line
            intra_line, = ax1.plot(range(1, len(intra_cleaned) + 1), intra_cleaned,
                                   label=f'Intra Depth {depth}', linestyle='-', color=colors[i])
            intra_lines.append(intra_line)

            # Plot inter with dashed line
            inter_line, = ax1.plot(range(1, len(inter_cleaned) + 1), inter_cleaned,
                                   label=f'Inter Depth {depth}', linestyle='--', color=colors[i])
            inter_lines.append(inter_line)

    # Add title and labels for similarities
    ax1.set_title('Intra and Inter Cluster Similarity and Test Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Similarity')

    # Plot test accuracy on a secondary y-axis (make sure to use test_acc_epochs)
    ax2 = ax1.twinx()
    test_acc = results['test_acc']
    ax2.plot(test_acc_epochs, test_acc, label='Test Accuracy', color='orange', linestyle='-.', linewidth=2)
    ax2.set_ylabel('Test Accuracy')

    # Show legend for test accuracy
    ax2.legend(loc='upper right')

    # Create a combined legend for intra and inter similarity with depth-specific colors
    custom_lines = [plt.Line2D([0], [0], color=color, lw=2, linestyle='-') for color in colors]
    custom_lines_inter = [plt.Line2D([0], [0], color=color, lw=2, linestyle='--') for color in colors]

    custom_legend = custom_lines + custom_lines_inter
    custom_labels = [f'Intra Depth {i + 1}' for i in range(len(colors))] + [f'Inter Depth {i + 1}' for i in
                                                                            range(len(colors))]

    ax1.legend(custom_legend, custom_labels, loc='upper left')

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

