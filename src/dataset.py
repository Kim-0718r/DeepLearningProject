import random
import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

try:
    from .config import Config
    from .transforms import get_eval_transform, get_train_transform
except ImportError:
    from config import Config
    from transforms import get_eval_transform, get_train_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(class_dir):
    class_dir = Path(class_dir)
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_dataset_split(raw_data_dir=None, output_dir=None, cfg=Config, overwrite=False):
    raw_data_dir = Path(raw_data_dir or cfg.RAW_DATA_DIR)
    output_dir = Path(output_dir or cfg.DATA_DIR)

    class_dirs = sorted(path for path in raw_data_dir.iterdir() if path.is_dir())

    if len(class_dirs) != cfg.NUM_CLASSES:
        raise ValueError(f"Expected {cfg.NUM_CLASSES} classes, but found {len(class_dirs)}.")

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    random.seed(cfg.SEED)
    split_counts = {
        "train": int(cfg.IMAGES_PER_CLASS * cfg.TRAIN_RATIO),
        "val": int(cfg.IMAGES_PER_CLASS * cfg.VAL_RATIO),
    }
    split_counts["test"] = cfg.IMAGES_PER_CLASS - split_counts["train"] - split_counts["val"]

    for class_dir in class_dirs:
        image_files = list_image_files(class_dir)

        if len(image_files) < cfg.IMAGES_PER_CLASS:
            raise ValueError(
                f"{class_dir.name} has {len(image_files)} images. "
                f"Need at least {cfg.IMAGES_PER_CLASS}."
            )

        selected = image_files[:]
        random.shuffle(selected)
        selected = selected[: cfg.IMAGES_PER_CLASS]

        train_end = split_counts["train"]
        val_end = train_end + split_counts["val"]

        split_files = {
            "train": selected[:train_end],
            "val": selected[train_end:val_end],
            "test": selected[val_end:],
        }

        for split_name, files in split_files.items():
            target_dir = output_dir / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for src_path in files:
                dst_path = target_dir / src_path.name
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)


def get_dataloaders(data_dir=None, cfg=Config):
    data_dir = Path(data_dir or cfg.DATA_DIR)

    train_dataset = ImageFolder(
        root=data_dir / "train",
        transform=get_train_transform(cfg),
    )
    val_dataset = ImageFolder(
        root=data_dir / "val",
        transform=get_eval_transform(cfg),
    )
    test_dataset = ImageFolder(
        root=data_dir / "test",
        transform=get_eval_transform(cfg),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
