import torch
import torch.nn as nn

from src.analysis import summarize_confusion_pairs
from src.config import Config, get_checkpoint_path
from src.dataset import get_dataloaders
from src.evaluate import classification_report, confusion_matrix, evaluate_model
from src.models import get_model
from src.train import build_optimizer, train_model
from src.utils import get_device, prepare_output_dirs, save_json, set_seed
from src.visualize import (
    confusion_matrix_path,
    history_path,
    plot_confusion_matrix,
    plot_history,
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
        )
        matrix = confusion_matrix(
            result["y_true"],
            result["y_pred"],
            num_classes=cfg.NUM_CLASSES,
        )

        all_results[model_name] = {
            "test_loss": result["test_loss"],
            "test_acc": result["test_acc"],
        }

        plot_history(history, history_path(model_name, cfg))
        plot_confusion_matrix(matrix, class_names, confusion_matrix_path(model_name, cfg))

        save_json(history, cfg.METRIC_DIR / f"{model_name}_history.json")
        save_json(
            classification_report(result["y_true"], result["y_pred"], class_names),
            cfg.METRIC_DIR / f"{model_name}_classification_report.json",
        )
        save_json(
            summarize_confusion_pairs(result["y_true"], result["y_pred"], class_names),
            cfg.METRIC_DIR / f"{model_name}_confusion_pairs.json",
        )

    plot_model_comparison(all_results, cfg.FIGURE_DIR / "model_comparison.png")
    save_json(all_results, cfg.METRIC_DIR / "model_comparison.json")

    return all_results


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    run_all_experiments()
