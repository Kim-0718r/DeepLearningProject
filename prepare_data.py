from src.config import Config
from src.data_preprocessing import prepare_data_from_zip_files
from src.utils import set_seed


def main():
    set_seed(Config.SEED)
    prepare_data_from_zip_files(Config)


if __name__ == "__main__":
    main()
