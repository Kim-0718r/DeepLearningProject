from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .config import Config
except ImportError:
    from config import Config


def _prepare_save_path(save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def plot_history(history, save_path):
    save_path = _prepare_save_path(save_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(matrix, class_names, save_path):
    save_path = _prepare_save_path(save_path)

    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            plt.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_model_comparison(results, save_path, metric_name="test_acc"):
    save_path = _prepare_save_path(save_path)

    model_names = list(results.keys())
    metric_values = [results[model_name][metric_name] for model_name in model_names]

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, metric_values)
    plt.ylim(0, 1)
    plt.ylabel(metric_name)
    plt.xticks(rotation=20, ha="right")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def history_path(model_name, cfg=Config):
    return cfg.FIGURE_DIR / f"{model_name}_history.png"


def confusion_matrix_path(model_name, cfg=Config):
    return cfg.FIGURE_DIR / f"{model_name}_confusion_matrix.png"
