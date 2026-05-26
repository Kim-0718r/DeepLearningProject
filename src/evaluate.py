import torch

try:
    from .config import Config
except ImportError:
    from config import Config


def get_predictions(model, data_loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


def evaluate_model(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return {
        "test_loss": running_loss / total,
        "test_acc": correct / total,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def confusion_matrix(y_true, y_pred, num_classes=Config.NUM_CLASSES):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1

    return matrix


def classification_report(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    report = {}

    for class_idx, class_name in enumerate(class_names):
        tp = matrix[class_idx][class_idx]
        fp = sum(matrix[row][class_idx] for row in range(len(class_names))) - tp
        fn = sum(matrix[class_idx][col] for col in range(len(class_names))) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[class_idx]),
        }

    return report
