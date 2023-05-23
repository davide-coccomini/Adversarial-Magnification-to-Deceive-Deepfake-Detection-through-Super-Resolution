from torchsr.models import edsr_baseline
import argparse
from PIL import Image
import cv2
import os
import torch
import numpy as np
from tqdm import tqdm
import random

def super_resolution(image, opt):
    model = edsr_baseline(scale=2, pretrained=True)
    model = model.eval()
    model = model.to(opt.gpu_id)
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float).unsqueeze(0).to(opt.gpu_id)

    sr_img = model(image)
    return sr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

def resize_image(image, dim):
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', default="../../deep_fakes/datasets/processed/crops_ff", type=str,
                        help='Images path')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU ID')
    parser.add_argument('--dataset', default=1, type=int,
                        help='Dataset to be processed (0: Openforensics; 1: FF++)')
    parser.add_argument('--mode', default=0, type=int,
                        help='Mode (0: Downscale; 1: Upscale)')
    parser.add_argument('--subsample', default=10, type=int,
                        help='How many images to pick from dataset')
    opt = parser.parse_args()
    print(opt)
    if opt.dataset == 0:
        images_folders = os.listdir(opt.images_folder)
        with tqdm(total=len(images_folders)) as pbar:
            for index, identifier in enumerate(images_folders):
                identifier_path = os.path.join(opt.images_folder, identifier)
                for image_name in os.listdir(identifier_path):
                    image_path = os.path.join(identifier_path, image_name)
                    image = resize_image(cv2.imread(image_path), (224, 224))
                    lr_image = resize_image(image, (int(image.shape[0]/2), int(image.shape[1]/2)))
                    hr_image = super_resolution(lr_image, opt)
                    output_path = image_path.replace("Train_Faces", "Train_Faces_Magnified")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    cv2.imwrite(output_path, hr_image)
                pbar.update()
    elif opt.dataset == 1:
        images_paths = []
        ff_methods = os.listdir(opt.images_folder)
        for ff_method in ff_methods:
            method_path = os.path.join(opt.images_folder, ff_method)
            videos = os.listdir(method_path)
            with tqdm(total=len(videos)) as pbar:
                for video in videos:
                    video_path = os.path.join(method_path, video)
                    images = os.listdir(video_path)
                    image_names = random.sample(images, opt.subsample)
                    for image_name in image_names:
                        image_path = os.path.join(video_path, image_name)
                        image = cv2.imread(image_path)
                        image = resize_image(image, (224, 224))
                        if opt.mode == 0:
                            output_path = image_path.replace("crops_ff", "crops_ff_minimized" + str(opt.subsample))
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            cv2.imwrite(output_path, image)

                            lr_image = resize_image(image, (int(image.shape[0]/2), int(image.shape[1]/2)))
                            hr_image = super_resolution(lr_image, opt)
                            output_path = image_path.replace("crops_ff", "crops_ff_minimized" + str(opt.subsample) + "_magnified")
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            
                            cv2.imwrite(output_path, hr_image)
                        else:

                            hr_image = super_resolution(image, opt)
                            lr_image = resize_image(hr_image, (int(hr_image.shape[0]/2), int(hr_image.shape[1]/2)))
                            output_path = image_path.replace("crops_ff", "crops_ff_minimized" + str(opt.subsample) + "_magnified_v2")
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            
                            cv2.imwrite(output_path, lr_image)

                    pbar.update()

if __name__ == "__main__":
    main()