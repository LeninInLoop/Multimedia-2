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

Result_dirs = {
    "results": "Results",
    "linear": "Results/Linear Quantization",
    "linear_hist": "Results/Linear Quantization Histograms",
    "optimal": "Results/Optimal Quantization",
    "optimal_hist": "Results/Optimal Quantization Histograms",
}

def main():
    # ----------------------- Load Images ------------------------
    images_array = {
        "lenna": ImageHelper.load_image(images["lenna"]),
        "baboon": ImageHelper.load_image(images["baboon"]),
        "goldhill": ImageHelper.load_image(images["goldhill"]),
        "barbara": ImageHelper.load_image(images["barbara"]),
    }
    for dir_path in Result_dirs.values(): os.makedirs(dir_path, exist_ok=True)

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
                image_8bit=img,
                bits_to_keep=bits_to_keep
            )

            ImageHelper.save_image(
                image_path=os.path.join(Result_dirs["linear"], f"{name}_quantized_linear_{bits_to_keep}bit.bmp"),
                image=quantized_image
            )

            Plotter.plot_and_save_histogram(
                image_name=name,
                bits_to_keep=bits_to_keep,
                image_array=img,
                centroids=centroids,
                boundaries=boundaries,
                num_levels=num_levels,
                quantizer_type="linear",
                result_dir=Result_dirs["linear_hist"]
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
                image_8bit=img,
                bits_to_keep=bits_to_keep
            )

            ImageHelper.save_image(
                image_path=os.path.join(Result_dirs["optimal"], f"{name}_quantized_optimal_{bits_to_keep}bit.bmp"),
                image=quantized_image
            )

            Plotter.plot_and_save_histogram(
                image_name=name,
                bits_to_keep=bits_to_keep,
                image_array=img,
                centroids=centroids,
                boundaries=boundaries,
                num_levels=num_levels,
                quantizer_type="optimal",
                result_dir=Result_dirs["optimal_hist"]
            )

            print(BColors.OK_BLUE + f"{name} optimal quantized to {bits_to_keep}-bit." + BColors.ENDC)

            sorted_centroids = np.sort(centroids)
            print(BColors.OK_CYAN + f"  Values (Centroids): {BColors.ENDC}\n  {[round(c, 2) for c in sorted_centroids]}")
            print(BColors.OK_CYAN + f"  Partitions (Boundaries): {BColors.ENDC}\n  {[round(b, 2) for b in boundaries]}")
        print()

    print(BColors.OK_GREEN + BColors.BOLD + 20 * "-" + " Optimum Quantization Completed. " + 20 * "-" + BColors.ENDC)

    # ----------------------- Calculate PSNR ------------------------
    print(BColors.HEADER + BColors.BOLD + "\nCalculating PSNR... \n" + BColors.ENDC)

    psnr_results = {"linear": {}, "optimal": {}}

    for name, original_img in images_array.items():
        psnr_results["linear"][name] = {}
        psnr_results["optimal"][name] = {}

        for bits_to_keep in [1, 2, 4]:

            # --- Linear PSNR ---
            linear_img_path = os.path.join(Result_dirs["linear"], f"{name}_quantized_linear_{bits_to_keep}bit.bmp")
            linear_img = ImageHelper.load_image(linear_img_path)

            linear_psnr = ImageHelper.calculate_psnr(original_img, linear_img)
            psnr_results["linear"][name][f"{bits_to_keep}bit"] = linear_psnr

            print(BColors.OK_BLUE + f"  {name} Linear {bits_to_keep}-bit PSNR: {linear_psnr:.2f} dB" + BColors.ENDC)


            # --- Optimal PSNR ---
            optimal_img_path = os.path.join(Result_dirs["optimal"], f"{name}_quantized_optimal_{bits_to_keep}bit.bmp")
            optimal_img = ImageHelper.load_image(optimal_img_path)

            optimal_psnr = ImageHelper.calculate_psnr(original_img, optimal_img)
            psnr_results["optimal"][name][f"{bits_to_keep}bit"] = optimal_psnr

            print(BColors.OK_BLUE + f"  {name} Optimal {bits_to_keep}-bit PSNR: {optimal_psnr:.2f} dB" + BColors.ENDC)

        print()

    print(BColors.HEADER + BColors.BOLD + "\nSaving PSNR results..." + BColors.ENDC)
    json_path = os.path.join(Result_dirs["results"], "psnr_results.json")

    JsonHelper.save_to_json(file_path=json_path, data=psnr_results)
    print(BColors.OK_GREEN + BColors.BOLD + f"Successfully saved PSNR results to {json_path}" + BColors.ENDC)



if __name__ == '__main__':
    main()