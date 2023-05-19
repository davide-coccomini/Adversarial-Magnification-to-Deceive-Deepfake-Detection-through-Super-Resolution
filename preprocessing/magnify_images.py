from torchsr.models import edsr_baseline
import argparse
from PIL import Image
import cv2
import os
import torch
import numpy as np
from tqdm import tqdm


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
    parser.add_argument('--images_folder', default="../../datasets/openforensics/Train_Faces", type=str,
                        help='Images path')
    parser.add_argument('--gpu_id', default=1, type=int,
                        help='GPU ID')
    opt = parser.parse_args()
    print(opt)
    
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


if __name__ == "__main__":
    main()