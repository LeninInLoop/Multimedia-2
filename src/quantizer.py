import numpy as np

class Quantizer:
    """
    A class holding static methods for linear and optimal
    (Lloyd-Max) image quantization.
    """

    @staticmethod
    def quantize_linear(image_8bit: np.ndarray, bits_to_keep: int) -> tuple:

        if bits_to_keep < 1 or bits_to_keep > 8:
            raise ValueError("Bits to keep must be between 1 and 8.")

        # 1. Calculate quantization levels
        shift = 8 - bits_to_keep
        quantized_levels = image_8bit >> shift

        # 2. Calculate values (centroids)
        num_levels = 2 ** bits_to_keep
        scale_factor = 255.0 / (num_levels - 1)
        centroids = (np.arange(num_levels) * scale_factor)

        # 3. Rescale image
        rescaled_image = (quantized_levels * scale_factor).astype(np.uint8)

        # 4. Calculate decision boundaries (partitions)
        bin_size = 256.0 / num_levels  # e.g., 256 / 16 = 16
        boundaries = (np.arange(1, num_levels) * bin_size) - 0.5

        return rescaled_image, centroids, boundaries

    @staticmethod
    def quantize_optimal(
            image_8bit: np.ndarray,
            bits_to_keep: int,
            max_iterations: int = 50
    ) -> tuple:

        num_levels = 2 ** bits_to_keep
        unique_levels, unique_counts = np.unique(image_8bit, return_counts=True)

        full_counts = np.zeros(256)
        full_counts[unique_levels] = unique_counts
        full_levels = np.arange(256)

        # Initialize centroids
        centroids = np.linspace(0, 255, num_levels, dtype=float)

        for _ in range(max_iterations):
            levels_1d = full_levels[:, np.newaxis]
            centroids_1d = centroids[np.newaxis, :]

            # --- Assignment Step ---
            distances = np.abs(levels_1d - centroids_1d)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)

            # --- Update Step ---
            for i in range(num_levels):
                mask = (labels == i)
                group_counts = full_counts[mask]

                if np.sum(group_counts) > 0:
                    group_levels = full_levels[mask]
                    new_centroids[i] = np.average(group_levels, weights=group_counts)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        # --- Map the image to the final centroids ---
        image_expanded = image_8bit[..., np.newaxis]
        centroids_1d = centroids[np.newaxis, np.newaxis, :]

        distances = np.abs(image_expanded - centroids_1d)
        final_labels = np.argmin(distances, axis=2)

        quantized_image = centroids[final_labels].astype(np.uint8)

        # --- Calculate final decision boundaries (partitions) ---
        sorted_centroids = np.sort(centroids)
        boundaries = (sorted_centroids[:-1] + sorted_centroids[1:]) / 2.0

        return quantized_image, centroids, boundaries