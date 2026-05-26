from pathlib import Path
import json

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, test_loader, criterion, device, class_names=None):
    """
    학습된 모델을 test_loader로 평가하는 함수

    반환값:
    - test_loss
    - test_acc
    - classification_report
    - confusion_matrix
    - y_true
    - y_pred
    """
    model.eval()

    running_loss = 0.0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            _, preds = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_loss = running_loss / total
    test_acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
    }

    return results


def print_evaluation_summary(results):
    """
    평가 결과를 콘솔에 간단히 출력하는 함수
    """
    print("===== Evaluation Summary =====")
    print(f"Test Loss     : {results['test_loss']:.4f}")
    print(f"Test Accuracy : {results['test_acc']:.4f}")
    print("==============================")


def save_metrics(results, save_path):
    """
    평가 결과를 JSON 파일로 저장하는 함수

    numpy array는 JSON 저장이 불가능하므로
    confusion_matrix는 list로 변환하여 저장한다.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    json_results = {
        "test_loss": round(float(results["test_loss"]), 4),
        "test_acc": round(float(results["test_acc"]), 4),
        "classification_report": results["classification_report"],
        "confusion_matrix": results["confusion_matrix"].tolist(),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4, ensure_ascii=False)

    print(f"[Saved] Metrics saved to: {save_path}")