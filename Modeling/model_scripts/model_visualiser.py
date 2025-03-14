import matplotlib.pyplot as plt


def plot_loss(train_loss, test_loss, title="Training vs Test Loss"):
    """
    Plots training and test loss over epochs.
    """
    epochs = list(range(1, len(train_loss) + 1))  

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', linestyle='-')
    plt.plot(epochs, test_loss, label='Test Loss', linestyle='-')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_log_scale(train_loss, test_loss, title="Training vs Test Loss (Log Scale)"):
    """
    Plots training and test loss over epochs with a logarithmic scale on y-axis.
    """
    epochs = list(range(1, len(train_loss) + 1))  

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', linestyle='-', color='blue')
    plt.plot(epochs, test_loss, label='Test Loss', linestyle='-', color='red')

    plt.yscale('log')  # Set log scale for y-axis
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)  # Grid for log scale
    plt.show()


