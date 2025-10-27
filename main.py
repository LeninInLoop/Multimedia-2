import os

from src import ImageHelper, BColors
from src import Quantizer

images = {
    "lenna": "images/lena_gray.bmp",
    "baboon": "images/baboon_gray.bmp",
    "goldhill": "images/goldhill_gray.bmp",
    "barbara": "images/barbara_gray.bmp",
}

def main():
    # ----------------------- Load Images ------------------------
    images_array = {
        "lenna": ImageHelper.load_image(images["lenna"]),
        "baboon": ImageHelper.load_image(images["baboon"]),
        "goldhill": ImageHelper.load_image(images["goldhill"]),
        "barbara": ImageHelper.load_image(images["barbara"]),
    }

    print(BColors.OK_CYAN + "Images Info:" + BColors.ENDC )
    for name, img in images_array.items():
        print(BColors.OK_BLUE + f"\t{name} image shape: {img.shape}" + BColors.ENDC )

    print(BColors.OK_GREEN + BColors.BOLD + 20 * "-" + " Images loaded " + 20 * "-" + BColors.ENDC)

    # -----------------------  Linear Quantization ------------------------
    print(BColors.HEADER + BColors.BOLD + "\nStarting Linear Quantization: \n" + BColors.ENDC)
    for name, img in images_array.items():
        for bits_to_keep in [1, 2, 4]:
            quantized_image = Quantizer.quantize_linear(
                image_8bit=img,
                bits_to_keep=bits_to_keep
            )
            ImageHelper.save_image(
                image_path=os.path.join("Results", f"{name}_quantized_{bits_to_keep}bit.bmp"),
                image=quantized_image
            )
            print(BColors.OK_BLUE + f"{name} quantized to {bits_to_keep}-bit." + BColors.ENDC)
        print()
    print(BColors.OK_GREEN + BColors.BOLD + 20 * "-" +  " Linear Quantization Completed. " + 20 * "-" + BColors.ENDC )








if __name__ == '__main__':
    main()