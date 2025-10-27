from matplotlib import pyplot as plt
import numpy as np, os
from .helper import BColors

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

        try:
            plt.figure(figsize=(10, 6))

            # 1. Calculate and plot the histogram (normalized)
            hist_values, bin_edges, _ = plt.hist(
                image_array.ravel(),
                bins=256,
                range=(0, 256),
                density=True,
                color='black'
            )

            # Redraw with a clean fill to match the example
            plt.clf()
            plt.fill_between(bin_edges[:-1], hist_values, color='black', step='post')
            plt.ylim(bottom=0)
            plt.xlim(0, 255)

            # 2. Plot Partitions (Boundaries)
            for b in boundaries:
                plt.axvline(x=b, color='red', linestyle='--')

            # 3. Plot Values (Centroids)
            for c in centroids:
                # Find the bin index for the centroid
                bin_index = int(np.floor(c))
                if bin_index == 256:  # Handle edge case
                    bin_index = 255
                y_val = hist_values[bin_index]
                plt.plot(c, y_val, 'yo', markersize=8, markeredgecolor='black')

            plt.title(
                f'{quantizer_type.capitalize()} Quantizer: {image_name} ({bits_to_keep}-bit / {num_levels} Levels)')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Normalized Count (Probability)')

            # 4. Save the plot
            plot_filename = os.path.join(result_dir, f"{image_name}_{quantizer_type}_{bits_to_keep}bit_histogram.png")
            plt.savefig(plot_filename)
            plt.close()

        except Exception as e:
            print(BColors.FAIL + f"  Failed to plot histogram for {image_name}: {e}" + BColors.ENDC)

