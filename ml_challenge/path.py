import os


def get_assets_path() -> str:
    return os.environ.get("ASSETS_PATH", "assets")

