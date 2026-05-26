import torch.nn as nn
from torchvision import models

try:
    from .config import Config
except ImportError:
    from config import Config


class ScratchCNN(nn.Module):
    """Simple CNN baseline trained from scratch for 10-class crop classification."""

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_scratch_cnn(num_classes=Config.NUM_CLASSES):
    return ScratchCNN(num_classes=num_classes)


def _replace_efficientnet_classifier(model, num_classes):
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_scratch(num_classes=Config.NUM_CLASSES):
    """EfficientNet-B0 architecture trained from random initialization."""
    model = models.efficientnet_b0(weights=None)
    return _replace_efficientnet_classifier(model, num_classes)


def build_efficientnet_feature_extractor(num_classes=Config.NUM_CLASSES):
    """ImageNet pretrained EfficientNet-B0 with frozen feature extractor."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    return _replace_efficientnet_classifier(model, num_classes)


def build_efficientnet_finetuning(num_classes=Config.NUM_CLASSES):
    """ImageNet pretrained EfficientNet-B0 with all layers trainable."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    for param in model.parameters():
        param.requires_grad = True

    return _replace_efficientnet_classifier(model, num_classes)


def get_model(model_name, num_classes=Config.NUM_CLASSES):
    model_builders = {
        "scratch_cnn": build_scratch_cnn,
        "efficientnet_scratch": build_efficientnet_scratch,
        "efficientnet_feature_extractor": build_efficientnet_feature_extractor,
        "efficientnet_finetuning": build_efficientnet_finetuning,
    }

    if model_name not in model_builders:
        available = ", ".join(model_builders.keys())
        raise ValueError(f"Unknown model_name: {model_name}. Available models: {available}")

    return model_builders[model_name](num_classes=num_classes)


def get_trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)

    return sum(param.numel() for param in model.parameters())


if __name__ == "__main__": # 모델 구현 테스트용 코드
    model = get_model("scratch_cnn", num_classes=Config.NUM_CLASSES)
    print(model)

    model = get_model("efficientnet_scratch", num_classes=Config.NUM_CLASSES)
    print(model.classifier)

    model = get_model("efficientnet_feature_extractor", num_classes=Config.NUM_CLASSES)
    print(model.classifier)

    model = get_model("efficientnet_finetuning", num_classes=Config.NUM_CLASSES)
    print(model.classifier)
