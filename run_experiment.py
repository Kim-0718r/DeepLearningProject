import torch
import torch.nn as nn

from src.analysis import get_top_misclassifications
from src.config import Config, get_checkpoint_path
from src.data_preprocessing import get_dataloaders
from src.evaluate import evaluate_model
from src.models import get_model
from src.train import build_optimizer, train_model
from src.utils import get_device, prepare_output_dirs, save_json, set_seed
from src.visualize import (
    plot_confusion_matrix,
    plot_all_training_curves,
    plot_model_comparison,
)


def run_all_experiments(cfg=Config):
    set_seed(cfg.SEED)
    prepare_output_dirs(cfg)

    device = get_device()
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg=cfg)

    all_results = {}

    for model_name in cfg.MODEL_NAMES:
        print(f"\n===== {model_name} =====")

        model = get_model(model_name=model_name, num_classes=cfg.NUM_CLASSES)
        optimizer = build_optimizer(model=model, model_name=model_name, cfg=cfg)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=cfg.EPOCHS,
            checkpoint_path=get_checkpoint_path(model_name, cfg),
        )

        result = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )
        matrix = result["confusion_matrix"]

        all_results[model_name] = {
            "test_loss": float(result["test_loss"]),
            "test_acc": float(result["test_acc"]),
        }

        plot_all_training_curves(history, model_name, cfg.FIGURE_DIR)
        plot_confusion_matrix(
            matrix,
            class_names,
            cfg.FIGURE_DIR / f"{model_name}_confusion_matrix.png",
        )

        save_json(history, cfg.METRIC_DIR / f"{model_name}_history.json")
        save_json(
            result["classification_report"],
            cfg.METRIC_DIR / f"{model_name}_classification_report.json",
        )
        save_json(
            get_top_misclassifications(matrix, class_names),
            cfg.METRIC_DIR / f"{model_name}_confusion_pairs.json",
        )

    model_names = list(all_results.keys())
    test_acc_values = [all_results[name]["test_acc"] for name in model_names]
    test_loss_values = [all_results[name]["test_loss"] for name in model_names]

    plot_model_comparison(
        model_names,
        test_acc_values,
        "Test Accuracy",
        cfg.FIGURE_DIR / "model_comparison_accuracy.png",
    )
    plot_model_comparison(
        model_names,
        test_loss_values,
        "Test Loss",
        cfg.FIGURE_DIR / "model_comparison_loss.png",
    )
    save_json(all_results, cfg.METRIC_DIR / "model_comparison.json")

    return all_results


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    run_all_experiments()
