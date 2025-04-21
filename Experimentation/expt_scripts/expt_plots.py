import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_threshold_vs_metrics(thresholds, accuracies, recalls, title='Threshold vs Recall'):
    """ Plots recall against thresholds. """
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, color="#4C72B0", linewidth=1.5)
    plt.plot(thresholds, accuracies, color="orange", linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Recall (%)', fontsize=14)
    plt.xticks(thresholds, rotation=45)
    plt.yticks([i for i in range(0,110,10)])
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.4)
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
    plt.title('Train Losses')
    plt.xlabel('Run')
    plt.ylabel('Loss')
    # plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Test Losses
    plt.figure(figsize=(8, 6))  # Create a new figure for test losses
    for model_name, losses in test_loss.items():
        avg_loss = np.mean(losses)
        plt.plot(range(1, len(losses) + 1), losses, label=f'{model_name} Test Loss')
    plt.title('Test Losses')
    plt.xlabel('Run')
    plt.ylabel('Loss')
    # plt.yscale("log")
    plt.legend()
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





