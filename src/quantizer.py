import numpy as np

class Quantizer:
    """
    A class holding static methods for linear, optimal (scalar),
    and LBG (vector) image quantization.
    """

    @staticmethod
    def quantize_linear(image_8bit: np.ndarray, bits_to_keep: int) -> tuple:
        """
        Quantizes an 8-bit (0-255) image to a lower bit depth
        using uniform (linear) quantization.
        """
        if bits_to_keep < 1 or bits_to_keep > 8:
            raise ValueError("Bits to keep must be between 1 and 8.")

        shift = 8 - bits_to_keep
        quantized_levels = image_8bit >> shift

        num_levels = 2 ** bits_to_keep
        scale_factor = 255.0 / (num_levels - 1)
        centroids = (np.arange(num_levels) * scale_factor)

        rescaled_image = (quantized_levels * scale_factor).astype(np.uint8)

        bin_size = 256.0 / num_levels
        boundaries = (np.arange(1, num_levels) * bin_size) - 0.5

        return rescaled_image, centroids, boundaries

    @staticmethod
    def quantize_optimal(
            image_8bit: np.ndarray,
            bits_to_keep: int,
            max_iterations: int = 50
    ) -> tuple:
        """
        Quantizes an 8-bit image to a specified number of bits
        using the optimal Lloyd-Max algorithm (1D k-means).
        """
        num_levels = 2 ** bits_to_keep
        unique_levels, unique_counts = np.unique(image_8bit, return_counts=True)

        full_counts = np.zeros(256)
        full_counts[unique_levels] = unique_counts
        full_levels = np.arange(256)

        centroids = np.linspace(0, 255, num_levels, dtype=float)

        for _ in range(max_iterations):
            levels_1d = full_levels[:, np.newaxis]
            centroids_1d = centroids[np.newaxis, :]

            distances = np.abs(levels_1d - centroids_1d)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)

            for i in range(num_levels):
                mask = (labels == i)
                group_counts = full_counts[mask]

                if np.sum(group_counts) > 0:
                    group_levels = full_levels[mask]
                    new_centroids[i] = np.average(group_levels, weights=group_counts)
                else:
                    new_centroids[i] = centroids[i] # Keep old centroid

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        image_expanded = image_8bit[..., np.newaxis]
        centroids_1d = centroids[np.newaxis, np.newaxis, :]

        distances = np.abs(image_expanded - centroids_1d)
        final_labels = np.argmin(distances, axis=2)

        quantized_image = centroids[final_labels].astype(np.uint8)

        sorted_centroids = np.sort(centroids)
        boundaries = (sorted_centroids[:-1] + sorted_centroids[1:]) / 2.0

        return quantized_image, centroids, boundaries

    @staticmethod
    def _image_to_vectors(img: np.ndarray, block_shape: tuple) -> np.ndarray:
        """Helper to split an image into non-overlapping blocks (vectors)."""
        h, w = img.shape
        bh, bw = block_shape

        h_trimmed = (h // bh) * bh
        w_trimmed = (w // bw) * bw
        img_trimmed = img[:h_trimmed, :w_trimmed]

        shape = (h_trimmed // bh, w_trimmed // bw, bh, bw)
        strides = (bh * img.strides[0], bw * img.strides[1], img.strides[0], img.strides[1])
        blocks = np.lib.stride_tricks.as_strided(img_trimmed, shape=shape, strides=strides)

        vectors = blocks.reshape(-1, bh * bw)
        return vectors.astype(np.float64)

    @staticmethod
    def _run_kmeans(vectors: np.ndarray, codebook: np.ndarray, max_iterations: int) -> tuple: # <-- Modified return
        """Helper to refine a codebook using k-means. Returns final codebook and labels."""
        num_centroids = len(codebook)
        labels = None # Initialize labels
        for _ in range(max_iterations):
            distances = np.linalg.norm(vectors[:, np.newaxis, :] - codebook[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1) # Assign final labels here

            new_codebook = np.zeros_like(codebook)

            for i in range(num_centroids):
                mask = (labels == i)
                if np.any(mask):
                    new_codebook[i] = np.mean(vectors[mask], axis=0)
                else:
                    # If a centroid becomes empty, reinitialize it to a random vector
                    new_codebook[i] = vectors[np.random.choice(len(vectors))]

            if np.allclose(codebook, new_codebook):
                break
            codebook = new_codebook
        return codebook, labels # Return labels as well

    @staticmethod
    def quantize_lbg(
            image_8bit: np.ndarray,
            target_codebook_size: int,
            block_shape: tuple = (2, 2),
            epsilon: float = 1.0, # Small perturbation for splitting
            max_iterations_kmeans: int = 20
    ) -> tuple: # <-- MODIFIED RETURN TYPE
        """
        Performs Vector Quantization using the Lindo-Buzo-Gray (LBG) algorithm.

        Returns:
            - reconstructed_image (np.ndarray): The new quantized image.
            - codebook (np.ndarray): The final codebook (shape: N, vec_dim).
            - vectors (np.ndarray): The original input vectors (shape: M, vec_dim).
            - labels (np.ndarray): The final assignment of each vector to a codeword index (shape: M,).
        """
        print(f"  Starting LBG... (Target Codebook: {target_codebook_size})")

        h, w = image_8bit.shape
        bh, bw = block_shape
        vectors = Quantizer._image_to_vectors(image_8bit, block_shape)

        if len(vectors) == 0:
             raise ValueError("Image dimensions are too small for the specified block shape.")

        # Initialize with the global mean
        initial_centroid = np.mean(vectors, axis=0)
        codebook = np.array([initial_centroid])
        labels = np.zeros(len(vectors), dtype=int) # Initial labels

        # LBG splitting process
        while len(codebook) < target_codebook_size:
            print(f"    Splitting codebook from {len(codebook)} -> {len(codebook) * 2}")
            # Split each centroid slightly
            split_vectors_list = []
            for c in codebook:
                split_vectors_list.append(c - epsilon)
                split_vectors_list.append(c + epsilon)
            codebook = np.array(split_vectors_list)

            # Refine the new codebook using k-means
            codebook, labels = Quantizer._run_kmeans(vectors, codebook, max_iterations_kmeans)

        print(f"  Final codebook generated with {len(codebook)} codewords.")

        # Use final labels and codebook for reconstruction
        print("  Reconstructing image from final codebook and labels...")
        quantized_vectors = codebook[labels]

        # Reshape vectors back into image blocks
        h_trimmed = (h // bh) * bh
        w_trimmed = (w // bw) * bw
        quantized_blocks = quantized_vectors.reshape(h_trimmed // bh, w_trimmed // bw, bh, bw)

        # Reconstruct the image from blocks
        reconstructed_image_trimmed = quantized_blocks.swapaxes(1, 2).reshape(h_trimmed, w_trimmed)

        # Create a full-size image and place the reconstructed part
        reconstructed_image_full = np.zeros_like(image_8bit)
        reconstructed_image_full[:h_trimmed, :w_trimmed] = reconstructed_image_trimmed

        return reconstructed_image_full.astype(np.uint8), codebook, vectors, labels