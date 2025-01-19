import matplotlib.pyplot as plt

def plot_losses(epoch_losses, title="Training Loss Over Epochs"):

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Loss', marker='o', linestyle='-', color='b')
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.show()


