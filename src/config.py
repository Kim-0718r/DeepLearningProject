from pathlib import Path


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    ZIP_SOURCE_DIR = PROJECT_ROOT / "AI_Project"
    EXTRACT_DIR = PROJECT_ROOT / "data" / "raw_extracted"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEST_DIR = DATA_DIR / "test"

    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    FIGURE_DIR = OUTPUT_DIR / "figures"
    METRIC_DIR = OUTPUT_DIR / "metrics"

    NUM_CLASSES = 10
    IMAGES_PER_CLASS = 500
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    EPOCHS = 20
    SEED = 42

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Four experiment conditions for the final report.
    MODEL_NAMES = [
        "scratch_cnn",
        "efficientnet_scratch",
        "efficientnet_feature_extractor",
        "efficientnet_finetuning",
    ]

    LEARNING_RATES = {
        "scratch_cnn": 1e-3,
        "efficientnet_scratch": 5e-4,
        "efficientnet_feature_extractor": 1e-3,
        "efficientnet_finetuning": 1e-4,
    }

    WEIGHT_DECAY = 1e-4
    CHECKPOINT_EXT = ".pt"


def get_checkpoint_path(model_name, cfg=Config):
    return cfg.CHECKPOINT_DIR / f"{model_name}{cfg.CHECKPOINT_EXT}"
