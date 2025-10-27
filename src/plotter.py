import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from src import BColors

class Plotter:

    @staticmethod
    def plot_and_save_histogram(
            image_name,
            bits_to_keep,
            image_array,
            centroids,
            boundaries,
            num_levels,
            quantizer_type,
            result_dir
    ):
        """
        Plots the histogram for scalar quantization with centroids and boundaries.
        """
        try:
            plt.figure(figsize=(10, 6))

            hist_values, bin_edges, _ = plt.hist(
                image_array.ravel(),
                bins=256,
                range=(0, 256),
                density=True,
                color='black'
            )

            plt.clf()
            plt.fill_between(bin_edges[:-1], hist_values, color='black', step='post')
            plt.ylim(bottom=0)
            plt.xlim(0, 255)

            for b in boundaries:
                plt.axvline(x=b, color='red', linestyle='--')

            for c in centroids:
                bin_index = int(np.floor(c))
                if bin_index >= 255: # Handle edge case c == 255
                    bin_index = 254
                y_val = hist_values[bin_index]
                plt.plot(c, y_val, 'yo', markersize=8, markeredgecolor='black')

            plt.title(f'{quantizer_type.capitalize()} Quantizer: {image_name} ({bits_to_keep}-bit / {num_levels} Levels)')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Normalized Count (Probability)')

            plot_filename = os.path.join(result_dir, f"{image_name}_{quantizer_type}_{bits_to_keep}bit_histogram.png")
            plt.savefig(plot_filename)
            plt.close()

        except Exception as e:
            print(BColors.FAIL + f"  Failed to plot histogram for {image_name} ({quantizer_type}): {e}" + BColors.ENDC)

    @staticmethod
    def plot_lbg_clusters(
            image_name: str,
            vectors: np.ndarray,
            codebook: np.ndarray,
            labels: np.ndarray,
            target_codebook_size: int,
            block_shape: tuple,
            result_dir: str
    ):
        """
        Performs PCA to reduce vectors and codebook to 2D and plots them,
        coloring vectors by their assigned cluster label to show partitions.
        """
        print(f"    Generating LBG cluster plot for {image_name}...")
        try:
            if vectors.shape[1] < 2:
                 print(BColors.WARNING + "  Cannot perform PCA with less than 2 dimensions. Skipping LBG plot." + BColors.ENDC)
                 return

            # Perform PCA to reduce dimensionality to 2
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors)
            codebook_2d = pca.transform(codebook) # Use the same transformation

            plt.figure(figsize=(10, 8))

            # --- MODIFICATION START ---
            # Plot the projected input vectors, colored by their label
            # Use a colormap for distinct colors
            num_clusters = len(codebook)
            cmap = plt.cm.get_cmap('viridis', num_clusters) # Or 'tab20', 'jet' etc.

            scatter = plt.scatter(
                vectors_2d[:, 0],
                vectors_2d[:, 1],
                c=labels,          # Color points based on cluster label
                cmap=cmap,         # Use the chosen colormap
                s=2,               # Smaller size for many points
                alpha=0.5,         # Transparency
                label='Input Vectors (Projected & Clustered)' # Modified Label
            )
            # --- MODIFICATION END ---


            # Plot the projected code vectors (centroids) - make them stand out
            plt.scatter(
                codebook_2d[:, 0],
                codebook_2d[:, 1],
                c='red',           # Keep centroids red
                s=80,              # Make centroids larger
                edgecolors='black', # Add black edge
                marker='X',        # Use a different marker (e.g., 'X')
                label='Code Vectors (Projected Centroids)' # Modified Label
             )


            plt.title(f'LBG Clusters (PCA Projection): {image_name} ({target_codebook_size} codewords, {block_shape[0]}x{block_shape[1]} blocks)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            plt.grid(True)

            # Optional: Add a color bar if using many clusters
            # if num_clusters > 10:
            #     plt.colorbar(scatter, label='Cluster Index')


            # Save the plot
            plot_filename = os.path.join(result_dir, f"{image_name}_lbg_{target_codebook_size}_clusters_pca_colored.png") # Added _colored
            plt.savefig(plot_filename)
            plt.close()
            print(f"      LBG cluster plot saved to {plot_filename}")

        except ImportError:
             print(BColors.FAIL + "  Scikit-learn is required for PCA plotting. Please install it (`pip install scikit-learn`). Skipping LBG plot." + BColors.ENDC)
        except Exception as e:
            print(BColors.FAIL + f"  Failed to plot LBG clusters for {image_name}: {e}" + BColors.ENDC)