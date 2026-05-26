import glob
import os
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from .config import Config
except ImportError:
    from config import Config


CLASS_MAPPING = {
    "apple": "Apple",
    "cabbage_green": "Cabbage",
    "cabbage_red": "Cabbage",
    "cabbage red": "Cabbage",
    "chinese-cabbage": "Chinese-cabbage",
    "garlic_uiseong": "Garlic",
    "mandarine_hallabong": "Mandarine",
    "mandarine_onjumilgam": "Mandarine",
    "onion_red": "Onion",
    "onion_white": "Onion",
    "pear_chuhwang": "Pear",
    "pear_singo": "Pear",
    "persimmon_bansi": "Persimmon",
    "persimmon_booyu": "Persimmon",
    "persimmon_deabong": "Persimmon",
    "potato_seolbong": "Potato",
    "potato_sumi": "Potato",
    "radish_winter-radish": "Radish",
}


def get_agricultural_transforms(image_size=Config.IMAGE_SIZE):
    base_resize_scale = int(image_size * (256 / 224))

    train_transform = transforms.Compose(
        [
            transforms.Resize(base_resize_scale),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=255),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.Resize(base_resize_scale),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ]
    )

    return train_transform, val_test_transform


def _is_valid_image(image_path):
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def extract_aihub_zip_files(zip_source_dir=None, extract_dir=None):
    zip_source_dir = Path(zip_source_dir or Config.ZIP_SOURCE_DIR)
    extract_dir = Path(extract_dir or Config.EXTRACT_DIR)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not zip_source_dir.exists():
        raise FileNotFoundError(f"Zip source directory not found: {zip_source_dir}")

    zip_files = [path for path in zip_source_dir.iterdir() if path.suffix.lower() == ".zip"]

    for zip_path in zip_files:
        zip_name_lower = zip_path.name.lower()
        matched_class = None

        for rule, main_class in CLASS_MAPPING.items():
            if rule in zip_name_lower:
                matched_class = main_class
                break

        if matched_class is None:
            print(f"[Skip] No class mapping for: {zip_path.name}")
            continue

        target_extract_path = extract_dir / f"{matched_class}_{zip_path.stem}"
        if target_extract_path.exists() and any(target_extract_path.iterdir()):
            print(f"[Exists] {zip_path.name}")
            continue

        print(f"[Extract] {zip_path.name}")
        target_extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_extract_path)


def build_agricultural_subset(extract_dir=None, subset_dir=None, cfg=Config):
    extract_dir = Path(extract_dir or cfg.EXTRACT_DIR)
    subset_dir = Path(subset_dir or cfg.DATA_DIR)

    if subset_dir.exists():
        shutil.rmtree(subset_dir)

    final_classes = sorted(set(CLASS_MAPPING.values()))

    for main_class in final_classes:
        subfolders = [
            folder
            for folder in os.listdir(extract_dir)
            if folder.startswith(main_class) and (extract_dir / folder).is_dir()
        ]

        if not subfolders:
            print(f"[Warning] No source folders for: {main_class}")
            continue

        num_subfolders = len(subfolders)
        base_quota = cfg.IMAGES_PER_CLASS // num_subfolders
        remainder = cfg.IMAGES_PER_CLASS % num_subfolders
        class_sampled_images = []

        for i, folder in enumerate(subfolders):
            folder_path = extract_dir / folder
            sub_images = []

            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                sub_images.extend(glob.glob(str(folder_path / "**" / ext), recursive=True))

            candidate_images = [
                img
                for img in sub_images
                if os.path.getsize(img) > 0 and _is_valid_image(img)
            ]

            quota = base_quota + (1 if i < remainder else 0)
            random.shuffle(candidate_images)
            class_sampled_images.extend(candidate_images[:quota])

        random.shuffle(class_sampled_images)

        train_bound = int(len(class_sampled_images) * cfg.TRAIN_RATIO)
        val_bound = int(len(class_sampled_images) * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))

        split_images = {
            "train": class_sampled_images[:train_bound],
            "val": class_sampled_images[train_bound:val_bound],
            "test": class_sampled_images[val_bound:],
        }

        for split, image_paths in split_images.items():
            split_target_dir = subset_dir / split / main_class
            split_target_dir.mkdir(parents=True, exist_ok=True)

            for img_path in image_paths:
                src_path = Path(img_path)
                dst_path = split_target_dir / src_path.name
                counter = 1
                while dst_path.exists():
                    dst_path = split_target_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
                    counter += 1
                shutil.copy2(src_path, dst_path)

        print(
            f"[{main_class}] "
            f"Train:{len(split_images['train'])}, "
            f"Val:{len(split_images['val'])}, "
            f"Test:{len(split_images['test'])}"
        )


def prepare_data_from_zip_files(cfg=Config):
    extract_aihub_zip_files(cfg.ZIP_SOURCE_DIR, cfg.EXTRACT_DIR)
    build_agricultural_subset(cfg.EXTRACT_DIR, cfg.DATA_DIR, cfg)


def get_dataloaders(data_dir=None, cfg=Config):
    data_dir = Path(data_dir or cfg.DATA_DIR)
    train_transform, val_test_transform = get_agricultural_transforms(cfg.IMAGE_SIZE)

    train_dataset = datasets.ImageFolder(root=data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=data_dir / "val", transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=data_dir / "test", transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Class encoding: {train_dataset.class_to_idx}")
    return train_loader, val_loader, test_loader, train_dataset.classes


if __name__ == "__main__":
    random.seed(Config.SEED)
    prepare_data_from_zip_files(Config)
    print(f"Processed dataset is ready: {Config.DATA_DIR}")
