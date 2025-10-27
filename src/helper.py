import numpy as np, os
from PIL import Image

class ImageHelper:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        if not os.path.isfile(image_path):
            raise FileNotFoundError
        return np.array(Image.open(image_path))

    @staticmethod
    def save_image(image_path: str, image: np.ndarray) -> None:
        return Image.fromarray(image).save(image_path)

class BColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
