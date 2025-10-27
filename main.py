import numpy as np, os
from src import ImageHelper, BColors
from src import Quantizer, Plotter
from src import JsonHelper

images = {
    "lenna": "images/lena_gray.bmp",
    "baboon": "images/baboon_gray.bmp",
    "goldhill": "images/goldhill_gray.bmp",
    "barbara": "images/barbara_gray.bmp",
}

result_dirs = {
    "results": "Results",
    "linear": "Results/Linear Quantization",
    "linear_hist": "Results/Linear Quantization Histograms",
    "optimal": "Results/Optimal Quantization",
    "optimal_hist": "Results/Optimal Quantization Histograms",
    "lbg": "Results/LBG Quantization",
    "lbg_plots": "Results/LBG Quantization Plots",
}

def main():
    # ----------------------- Load Images ------------------------
    images_array = {
        "lenna": ImageHelper.load_image(images["lenna"]),
        "baboon": ImageHelper.load_image(images["baboon"]),
        "goldhill": ImageHelper.load_image(images["goldhill"]),
        "barbara": ImageHelper.load_image(images["barbara"]),
    }
    for dir_path in result_dirs.values(): os.makedirs(dir_path, exist_ok=True)

    print(BColors.OK_CYAN + "Images Info:" + BColors.ENDC)
    for name, img in images_array.items():
        print(BColors.OK_BLUE + f"\t{name} image shape: {img.shape}" + BColors.ENDC)

    print(BColors.OK_GREEN + BColors.BOLD + 20 * "-" + " Images loaded " + 20 * "-" + BColors.ENDC)

    # -----------------------  Linear Quantization ------------------------
    print(BColors.HEADER + BColors.BOLD + "\nStarting Linear Quantization: \n" + BColors.ENDC)
    for name, img in images_array.items():
        for bits_to_keep in [1, 2, 4]:
            num_levels = 2 ** bits_to_keep

            quantized_image, centroids, boundaries = Quantizer.quantize_linear(
                image_8bit=img, bits_to_keep=bits_to_keep
            )

            ImageHelper.save_image(
                image_path=os.path.join(result_dirs["linear"], f"{name}_quantized_linear_{bits_to_keep}bit.bmp"),
                image=quantized_image
            )

            Plotter.plot_and_save_histogram(
                image_name=name, bits_to_keep=bits_to_keep, image_array=img,
                centroids=centroids, boundaries=boundaries, num_levels=num_levels,
                quantizer_type="linear", result_dir=result_dirs["linear_hist"]
            )

            print(BColors.OK_BLUE + f"{name} linear quantized to {bits_to_keep}-bit." + BColors.ENDC)
            sorted_centroids = np.sort(centroids)

            print(BColors.OK_CYAN + f"  Values (Centroids): {BColors.ENDC}\n  {[round(c, 2) for c in sorted_centroids]}")
            print(BColors.OK_CYAN + f"  Partitions (Boundaries): {BColors.ENDC}\n  {[round(b, 2) for b in boundaries]}")

        print()
    print(BColors.OK_GREEN + BColors.BOLD + 20 * "-" + " Linear Quantization Completed. " + 20 * "-" + BColors.ENDC)

    # -----------------------  Optimum Quantization ------------------------
    print(BColors.HEADER + BColors.BOLD + "\nStarting Optimum Quantization: \n" + BColors.ENDC)
    for name, img in images_array.items():
        for bits_to_keep in [1, 2, 4]:
            num_levels = 2 ** bits_to_keep

            quantized_image, centroids, boundaries = Quantizer.quantize_optimal(
                image_8bit=img, bits_to_keep=bits_to_keep
            )

            ImageHelper.save_image(
                image_path=os.path.join(result_dirs["optimal"], f"{name}_quantized_optimal_{bits_to_keep}bit.bmp"),
                image=quantized_image
            )

            Plotter.plot_and_save_histogram(
                image_name=name, bits_to_keep=bits_to_keep, image_array=img,
                centroids=centroids, boundaries=boundaries, num_levels=num_levels,
                quantizer_type="optimal", result_dir=result_dirs["optimal_hist"]
            )

            print(BColors.OK_BLUE + f"{name} optimal quantized to {bits_to_keep}-bit." + BColors.ENDC)
            sorted_centroids = np.sort(centroids)

            print(BColors.OK_CYAN + f"  Values (Centroids): {BColors.ENDC}\n  {[round(c, 2) for c in sorted_centroids]}")
            print(BColors.OK_CYAN + f"  Partitions (Boundaries): {BColors.ENDC}\n  {[round(b, 2) for b in boundaries]}")

        print()
    print(BColors.OK_GREEN + BColors.BOLD + 20 * "-" + " Optimum Quantization Completed. " + 20 * "-" + BColors.ENDC)

    # ----------------------- LBG Vector Quantization ------------------------
    print(BColors.HEADER + BColors.BOLD + "\nStarting LBG Vector Quantization (Task 4): \n" + BColors.ENDC)

    target_codebook_size = 32
    block_shape = (2, 2)
    lbg_quantized_images = {}

    for name, img in images_array.items():
        print(BColors.OK_CYAN + f"Processing LBG for {name}..." + BColors.ENDC)

        quantized_lbg_image, lbg_codebook, lbg_vectors, lbg_labels = Quantizer.quantize_lbg(
            image_8bit=img, # Use current image
            target_codebook_size=target_codebook_size,
            block_shape=block_shape
        )

        lbg_quantized_images[name] = quantized_lbg_image

        lbg_save_path = os.path.join(result_dirs["lbg"], f"{name}_lbg_{target_codebook_size}_codewords.bmp")
        ImageHelper.save_image(image_path=lbg_save_path, image=quantized_lbg_image)
        print(BColors.OK_BLUE + f"  {name} LBG quantized image saved to {lbg_save_path}" + BColors.ENDC)

        # Save Codebook and details to JSON
        lbg_details_data = {
            "image": name,
            "target_codebook_size": target_codebook_size,
            "block_shape": list(block_shape),
            "vector_dimension": block_shape[0] * block_shape[1],
            "final_codebook_size": len(lbg_codebook),
            "codebook": lbg_codebook.tolist()
        }

        lbg_json_path = os.path.join(result_dirs["lbg"], f"{name}_lbg_{target_codebook_size}_details.json")
        JsonHelper.save_to_json(file_path=lbg_json_path, data=lbg_details_data)
        print(BColors.OK_BLUE + f"  {name} LBG codebook and details saved to {lbg_json_path}" + BColors.ENDC)

        # Call LBG plotting function
        Plotter.plot_lbg_clusters(
            image_name=name,
            vectors=lbg_vectors,
            codebook=lbg_codebook,
            labels=lbg_labels,
            target_codebook_size=target_codebook_size,
            block_shape=block_shape,
            result_dir=result_dirs["lbg_plots"]
        )
        print()

    print(BColors.OK_GREEN + BColors.BOLD + 20 * "-" + " LBG Quantization Completed. " + 20 * "-" + BColors.ENDC)


    # ----------------------- Calculate PSNR ------------------------
    print(BColors.HEADER + BColors.BOLD + "\nCalculating PSNR... \n" + BColors.ENDC)

    psnr_results = {"linear": {}, "optimal": {}, "lbg": {}}

    for name, original_img in images_array.items():
        psnr_results["linear"][name] = {}
        psnr_results["optimal"][name] = {}
        psnr_results["lbg"][name] = {}

        for bits_to_keep in [1, 2, 4]:

            # --- Linear PSNR ---
            linear_img_path = os.path.join(result_dirs["linear"], f"{name}_quantized_linear_{bits_to_keep}bit.bmp")
            linear_img = ImageHelper.load_image(linear_img_path)

            linear_psnr = ImageHelper.calculate_psnr(original_img, linear_img)
            psnr_results["linear"][name][f"{bits_to_keep}bit"] = linear_psnr

            print(BColors.OK_BLUE + f"  {name} Linear {bits_to_keep}-bit PSNR: {linear_psnr:.2f} dB" + BColors.ENDC)

            # --- Optimal PSNR ---
            optimal_img_path = os.path.join(result_dirs["optimal"], f"{name}_quantized_optimal_{bits_to_keep}bit.bmp")
            optimal_img = ImageHelper.load_image(optimal_img_path)

            optimal_psnr = ImageHelper.calculate_psnr(original_img, optimal_img)
            psnr_results["optimal"][name][f"{bits_to_keep}bit"] = optimal_psnr

            print(BColors.OK_BLUE + f"  {name} Optimal {bits_to_keep}-bit PSNR: {optimal_psnr:.2f} dB" + BColors.ENDC)

        # --- LBG PSNR ---
        quantized_lbg_img = lbg_quantized_images.get(name)
        lbg_psnr = ImageHelper.calculate_psnr(original_img, quantized_lbg_img)

        psnr_results["lbg"][name][f"{target_codebook_size}_codewords_{block_shape[0]}x{block_shape[1]}"] = lbg_psnr
        print(BColors.OK_CYAN + f"  {name} LBG {target_codebook_size}-codeword ({block_shape[0]}x{block_shape[1]}) PSNR: {lbg_psnr:.2f} dB" + BColors.ENDC)
        print()

    print(BColors.HEADER + BColors.BOLD + "\nSaving PSNR results..." + BColors.ENDC)
    json_path = os.path.join(result_dirs["results"], "psnr_results.json")

    JsonHelper.save_to_json(file_path=json_path, data=psnr_results)
    print(BColors.OK_GREEN + BColors.BOLD + f"Successfully saved PSNR results to {json_path}" + BColors.ENDC)


if __name__ == '__main__':
    main()