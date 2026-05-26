from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curve(history, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Loss curve saved to: {save_path}")


def plot_accuracy_curve(history, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Accuracy curve saved to: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path, normalize=False):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = format(cm[i, j], fmt)
            plt.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Confusion matrix saved to: {save_path}")


def plot_model_comparison(model_names, metric_values, metric_name, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, metric_values)

    plt.title(f"Model Comparison - {metric_name}")
    plt.xlabel("Model")
    plt.ylabel(metric_name)

    for i, value in enumerate(metric_values):
        plt.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Saved] Model comparison chart saved to: {save_path}")


def plot_all_training_curves(history, model_name, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curve(
        history=history,
        save_path=save_dir / f"{model_name}_loss_curve.png",
    )

    plot_accuracy_curve(
        history=history,
        save_path=save_dir / f"{model_name}_accuracy_curve.png",
    )