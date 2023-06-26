from math import log10, sqrt
import cv2
import numpy as np
import os
import argparse
from statistics import mean
from skimage.metrics import structural_similarity

def resize_image(image, dim):
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_images_folder', default="../../deep_fakes/datasets/processed/crops_ff_minimized10", type=str,
                        help='Images path')
    parser.add_argument('--edited_images_folder', default="../../deep_fakes/datasets/processed/crops_ff_minimized10_magnified_v3_scale2", type=str,
                        help='Images path')
    parser.add_argument('--metric', default=0, type=int,
                        help='Metric (0: psnr; 1: SSIM)')
    opt = parser.parse_args()
    print(opt)

    for method in os.listdir(opt.original_images_folder):
        method_path = os.path.join(opt.original_images_folder, method)
        values = []
        diffs = []
        for index, video in enumerate(os.listdir(method_path)):
            video_path = os.path.join(method_path, video)
            image_name = os.listdir(video_path)[0]
            original_image_path = os.path.join(video_path, image_name)
            edited_image_path = original_image_path.replace(opt.original_images_folder, opt.edited_images_folder)
            original = cv2.imread(original_image_path)
            edited = resize_image(cv2.imread(edited_image_path, 1), (224, 224))
            if opt.metric == 0:

                value = PSNR(original, edited)
                values.append(value)
            else:
                (score, diff) = structural_similarity(original, edited, full=True, multichannel=True)
                values.append(score)
                diffs.append(diff)

        print("Method ", method, "FINAL SCORE: ", mean(values), " DIFFS:", mean(diffs))

if __name__ == "__main__":
    main()