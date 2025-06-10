import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from math import pi
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.colors as mcolors
import math

def plot_threshold_vs_metrics(thresholds, accuracies, recalls, title='Threshold vs Recall'):
    """ Plots recall against thresholds. """
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds[1:], recalls[1:], label='Recall', color="#4C72B0", linewidth=1.5)
    plt.plot(thresholds[1:], accuracies[1:], label='Accuracy', color="orange", linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Accuracy and Recall (%)', fontsize=14)
    plt.xticks(thresholds, rotation=45, fontsize=10)
    plt.yticks([i for i in range(0,110,10)], fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_acc_vs_recall(thresholds, accuracies, recalls, title='Threshold vs Recall'):

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(recalls[1:], accuracies[1:], label='Precision vs Recall', color="#4C72B0", linewidth=1.5)
    plt.scatter(recalls[1:], accuracies[1:], color="red", s=40)

    for i in range(1, len(thresholds)):
        plt.annotate(f'{thresholds[i]:.2f}',
                     (recalls[i], accuracies[i]),
                     textcoords="offset points",
                     xytext=(0, 6),
                     ha='center',
                     fontsize=10,
                     color='black')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Recall (%)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_acc_vs_recall_for_paper(thresholds, accuracies_dict, recalls_dict, title=''):
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    key_indices = [1,6,10]  # just a few labels
    
    for model_name in accuracies_dict:
        acc = accuracies_dict[model_name]
        rec = recalls_dict[model_name]
        
        plt.plot(rec[1:], acc[1:], label=model_name, linewidth=1.5)

        for i in key_indices:
            plt.scatter(rec[i], acc[i], s=30, color='gray', zorder=3)
            plt.annotate(f'{thresholds[i]:.1f}',
                        (rec[i], acc[i]),
                        textcoords="offset points",
                        xytext=(0, -25),
                        ha='center',
                        fontsize=12,
                        color='black')
    
    # plt.title(title, fontsize=16)
    plt.xlabel('Recall (%)', fontsize=16)
    plt.ylabel('Precision (%)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    # Add dummy scatter for threshold marker legend
    plt.scatter([], [], color='gray', s=30, label='Threshold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()




def plot_accuracies(thresholds, accuracies_dict, title='Threshold vs Accuracies'):
    """ Plots accuracy (%) vs thresholds for multiple models.
    """
    plt.figure(figsize=(10, 6))
    for model_name, acc_values in accuracies_dict.items():
        plt.plot(thresholds, acc_values, label=model_name, linewidth=1.5)

    plt.title(title, fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(thresholds)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.show()


def plot_recalls(thresholds, recalls_dict, title='Threshold vs Recalls'):
    """ Plots recall (%) vs thresholds for multiple models.
    """
    plt.figure(figsize=(10, 6))
    for model_name, recall_values in recalls_dict.items():
        plt.plot(thresholds, recall_values, label=model_name, linewidth=1.5)

    plt.title(title, fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Recall (%)', fontsize=14)
    plt.xticks(thresholds)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Models')
    plt.tight_layout()
    plt.show()


def load_avg_losses(model_names, output_dir="Results"):
    """ Load average train and test losses from JSON files for all models.
    """
    train_loss = {}
    test_loss = {}

    for model_name in model_names:
        avg_file = os.path.join(output_dir, f"{model_name}_avg.json")
        with open(avg_file, "r") as f:
            data = json.load(f)
            # Extract average losses from the JSON file
            train_loss[model_name] = data["avg_train_loss"]
            test_loss[model_name] = data["avg_test_loss"]

    return train_loss, test_loss



def plot_losses_ae(train_loss, test_loss):
    """ Plot train losses and test losses for all models in separate plots.
    """
    # Plot Train Losses
    plt.figure(figsize=(8, 6))  # Create a new figure for train losses
    for model_name, losses in train_loss.items():
        avg_loss = np.mean(losses)
        plt.plot(range(1, len(losses) + 1), losses, label=f'{model_name} Train Loss')
    # plt.title('Train Losses')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.yscale("log")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot Test Losses
    plt.figure(figsize=(8, 6))  # Create a new figure for test losses
    for model_name, losses in test_loss.items():
        avg_loss = np.mean(losses)
        plt.plot(range(1, len(losses) + 1), losses, label=f'{model_name} Test Loss')
    # plt.title('Test Losses')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    # plt.yscale("log")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_losses_from_json(json_path, title="Train vs Test Loss"):
    """ Given a JSON file path, this function reads the train and test losses and plots them together in a single plot.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Try keys depending on whether it's a run file or avg file
    train_losses = data.get("train_losses", data.get("avg_train_loss"))
    test_losses = data.get("test_losses", data.get("avg_test_loss"))

    if train_losses is None or test_losses is None:
        print(f"Missing losses in: {json_path}")
        return

    model_name = os.path.basename(json_path).replace(".json", "")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(test_losses)+1), test_losses, label="Test Loss", linestyle='-')
    plt.title(title)
    plt.xlabel("Epoch" if "avg" in json_path else "Run")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return train_losses, test_losses


# code: 19 may 2025
def plot_pretty_radar(metrics, save_path="pretty_radar_plot.png"):

    df = pd.DataFrame(metrics).T / 100
    labels = df.columns.tolist()
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    plt.style.use("seaborn-v0_8-muted")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    for i, (model, row) in enumerate(df.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, label=model, linewidth=2.5, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10, color="gray")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7)

    plt.title("Radar Plot of Model Metrics", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    num_models = len(data)
    cols = math.ceil(math.sqrt(num_models))
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (model_name, preds) in enumerate(data.items()):
        y_true = preds['y_true']
        y_pred = preds['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], colorbar=False)
        axes[i].set_title(model_name)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()



def plot_confusion_matrices(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_models = len(data)
    # cols = math.ceil(math.sqrt(num_models))
    # rows = math.ceil(num_models / cols)
    cols = num_models
    rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    cell_colors = {
        (0, 0): '#C8E6C9',  # TN
        (0, 1): '#FFCDD2',  # FP
        (1, 0): '#FFF9C4',  # FN
        (1, 1): '#BBDEFB',  # TP
    }

    for i, (model_name, preds) in enumerate(data.items()):
        y_true = preds['y_true']
        y_pred = preds['y_pred']
        cm = confusion_matrix(y_true, y_pred)

        ax = axes[i]
        # for (row, col), color in cell_colors.items():
        #     ax.fill_between([col, col+1], row, row+1, color=color, edgecolor=None)

        #     count = cm[row, col]
        #     ax.text(col + 0.5, row + 0.5, str(count), ha='center', va='center',
        #             fontsize=14, fontweight='bold', color='black')

        ax.set_title(model_name, fontsize=20, pad=10)
        ax.set_xticks(np.arange(2) + 0.5)
        ax.set_yticks(np.arange(2) + 0.5)
        ax.set_xticklabels(['Pred 0', 'Pred 1'], rotation=0, fontsize=16)
        ax.set_yticklabels(['True 0', 'True 1'], rotation=90, fontsize=16)
        ax.invert_yaxis()
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Draw cells with labels
        for row in range(2):
            for col in range(2):
                ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                           facecolor=cell_colors[(row, col)],
                                           edgecolor='white', lw=2))
                value = cm[row, col]
                ax.text(col + 0.5, row + 0.5, str(value),
                        ha='center', va='center',
                        fontsize=20, fontweight='bold', color='black')

        ax.set_title(model_name, fontsize=20, pad=10)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


