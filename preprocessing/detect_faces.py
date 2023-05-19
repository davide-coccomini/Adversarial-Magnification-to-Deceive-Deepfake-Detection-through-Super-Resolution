import argparse
import json
import os
import numpy as np
from typing import Type
import cv2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pandas as pd
import face_detector
from face_detector import ImageDataset
from face_detector import ImageFaceDetector
import argparse
from PIL import Image

def process_images(images, opt):
    detector = ImageFaceDetector(device=opt.gpu_id, thresholds=[0.8, 0.8, 0.8])
    dataset = ImageDataset(images)

    loader = DataLoader(dataset, shuffle=False, num_workers=40, batch_size=1, collate_fn=lambda x: x)
    missed_images = []
    for item in tqdm(loader): 
        item = item[0]
        id = item[0]
        image_path = item[1]
        image = Image.open(image_path)

        out_dir = os.path.join(opt.output_path, str(id))

        faces_bboxes = detector._detect_faces(image)
        
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for j, bbox in enumerate(faces_bboxes):
            xmin, ymin, xmax, ymax = [int(b) for b in bbox]

            w = xmax - xmin
            h = ymax - ymin

            p_h = int(h * 0.10)
            p_w = int(w * 0.10)
            
            crop_h = (ymax + p_h) - max(ymin - p_h, 0)
            crop_w = (xmax + p_w) - max(xmin - p_w, 0)

            
            # Make the image square
            if crop_h > crop_w:
                p_h -= int(((crop_h - crop_w)/2))
            else:
                p_w -= int(((crop_w - crop_h)/2))

            crop_h = (ymax + p_h) - max(ymin - p_h, 0)
            crop_w = (xmax + p_w) - max(xmin - p_w, 0)

            crop = image[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "{}.png".format(j)), crop)
        
        



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', default="../../datasets/openforensics/Train_poly.json", type=str,
                        help='Images List file path')
    parser.add_argument('--data_path', type=str,
                        help='Data directory', default='../../datasets/openforensics')
    parser.add_argument('--output_path', type=str,
                        help='Output directory', default='../../datasets/openforensics/Train_Faces')
    parser.add_argument('--gpu_id', type=int,
                        help='GPU ID', default=0)

                        
    opt = parser.parse_args()
    print(opt)
    f = open(opt.list_file)
    images_json = dict(json.load(f))["images"]
    images_paths = []
    for image in images_json:
        images_paths.append((image["id"], os.path.join(opt.data_path, image["file_name"].replace("Images/", ""))))
    process_images(images_paths, opt)
   
if __name__ == "__main__":
    main()