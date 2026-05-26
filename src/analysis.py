from pathlib import Path

import torch
from torchvision.utils import save_image

try:
    from .config import Config
except ImportError:
    from config import Config


def summarize_confusion_pairs(y_true, y_pred, class_names, top_k=5):
    pair_counts = {}

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            continue

        pair = (class_names[true_label], class_names[pred_label])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)

    return [
        {
            "true_class": pair[0],
            "pred_class": pair[1],
            "count": count,
        }
        for pair, count in sorted_pairs[:top_k]
    ]


def collect_misclassified_samples(model, data_loader, class_names, device, max_samples=20):
    model.eval()
    samples = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            for image, true_label, pred_label in zip(images.cpu(), labels.cpu(), preds.cpu()):
                if true_label.item() == pred_label.item():
                    continue

                samples.append(
                    {
                        "image": image,
                        "true_class": class_names[true_label.item()],
                        "pred_class": class_names[pred_label.item()],
                    }
                )

                if len(samples) >= max_samples:
                    return samples

    return samples


def save_misclassified_samples(samples, output_dir=None, cfg=Config):
    output_dir = Path(output_dir or cfg.FIGURE_DIR / "misclassified")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for idx, sample in enumerate(samples, start=1):
        file_name = f"{idx:02d}_true-{sample['true_class']}_pred-{sample['pred_class']}.png"
        save_path = output_dir / file_name
        save_image(sample["image"], save_path)
        saved.append(str(save_path))

    return saved
