from torchvision import transforms

try:
    from .config import Config
except ImportError:
    from config import Config


def get_train_transform(cfg=Config):
    return transforms.Compose(
        [
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]
    )


def get_eval_transform(cfg=Config):
    return transforms.Compose(
        [
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
        ]
    )
