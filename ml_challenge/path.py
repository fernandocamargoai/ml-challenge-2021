import os


def get_assets_path() -> str:
    if "TRAINML_DATA_PATH" in os.environ:
        return os.path.join(os.environ["TRAINML_DATA_PATH"], "assets")
    else:
        return "assets"
