import numpy as np

class Quantizer:

    @staticmethod
    def quantize_linear(image_8bit: np.ndarray, bits_to_keep: int) -> np.ndarray:
        if bits_to_keep < 1 or bits_to_keep > 8:
            raise ValueError("Bits to keep must be between 1 and 8.")

        shift = 8 - bits_to_keep
        quantized_levels = image_8bit >> shift

        num_levels = 2 ** bits_to_keep
        scale_factor = 255.0 / (num_levels - 1)

        return (quantized_levels * scale_factor).astype(np.uint8)
