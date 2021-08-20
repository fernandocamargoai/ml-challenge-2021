import os


def get_assets_path() -> str:
    return os.environ.get("ASSETS_PATH", "assets")


def get_extra_data_path() -> str:
    return os.environ.get("EXTRA_DATA_PATH", "extra_data")
