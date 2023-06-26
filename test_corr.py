import torch
import random
import yaml
import argparse
import pandas as pd
import os
from os import cpu_count
import cv2
import numpy as np
import math
from sklearn.model_selection import train_test_split
import collections
from deepfakes_dataset import DeepFakesDataset
from torchvision.models import resnet50, ResNet50_Weights
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from progress.bar import ChargingBar
from utils import check_correct, unix_time_millis
from timm.scheduler.cosine_lr import CosineLRScheduler
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.metrics import f1_score, confusion_matrix
from skimage.metrics import structural_similarity

from transformers import ViTForImageClassification, ViTConfig
import timm


def convert_list_to_string(lst):
    return ' '.join(str(element) for element in lst)

# Main body
if __name__ == "__main__":
    
    random.seed(42)
    torch.manual_seed(43)  
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--fake_data_path', default='../deep_fakes/datasets/processed/crops_ff_minimized10', type=str,
                        help='Videos directory')
    parser.add_argument('--pristine_data_path', default='../deep_fakes/datasets/processed/crops_ff_minimized10', type=str,
                        help='Videos directory')
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for validation (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--list_file', default="../deep_fakes/datasets/test_videos.csv", type=str,
                        help='Images List txt file path)')
    parser.add_argument('--use_pretrained', type=bool, default=True, 
                        help="Use pretrained models")
    parser.add_argument('--dataset', default=1, type=int,
                        help='Dataset to be processed (0: Openforensics; 1: FF++)')
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained")
    parser.add_argument('--max_images', type=int, default=-1, 
                        help="Maximum number of images to use for training (default: all).")
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--forgery_method', type=str, default='',
                        help="")
    opt = parser.parse_args()
    print(opt)

    if opt.config != '':
        with open(opt.config, 'r') as ymlfile:
            config = yaml.safe_load(ymlfile)


    if opt.model == 0: 
        HUB_URL = "SharanSMenon/swin-transformer-hub:main"
        MODEL_NAME = "swin_tiny_patch4_window7_224"
        model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)
        model.head = torch.nn.Linear(768, config['model']['num-classes'])
    elif opt.model == 1: 
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(2048, config['model']['num-classes'])
    elif opt.model == 2:
        model = timm.create_model('xception', pretrained=True, num_classes = config['model']['num-classes'])
    elif opt.model == 3:
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=config['model']['num-classes'])

    if opt.model_path != '':
        model_path = opt.model_path
        while not os.path.exists(model_path):
            epoch = int(model_path.split("_")[-1].replace("checkpoint", ""))
            new_epoch = epoch - 1
            model_path = model_path.replace(str(epoch), str(new_epoch))
            print("Trying new model weights", model_path)
            if new_epoch == 0:
                print("No model found.")
                exit()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0')))
        print("Weights loaded", model_path)
    else:
        print("No weights loaded.")
        exit()

    

    model = model.to(opt.gpu_id)
    model.eval()

    loss_fn = torch.nn.BCEWithLogitsLoss()


    if opt.dataset == 1:
        images_paths = []
        test_labels = []
        df_test = pd.read_csv(opt.list_file, names=["video", "label"], sep=" ")


        for index, row in df_test.iterrows():
            if opt.forgery_method in row["video"]:
                video_path = os.path.join(opt.fake_data_path, row["video"])
            elif "Original" in row["video"]:
                video_path = os.path.join(opt.pristine_data_path, row["video"])
            else:
                continue
        
            for image_name in os.listdir(video_path):
                image_path = os.path.join(video_path, image_name)
                images_paths.append(image_path)
                test_labels.append(row["label"])

    
    if opt.max_images > -1:
        images_paths = images_paths[:opt.max_images]
        test_labels = test_labels[:opt.max_images]


    test_dataset = DeepFakesDataset(images_paths, test_labels, mode='val', additional_path = ["crops_ff_minimized10", "crops_ff_minimized10_magnified_scale4"])
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)

                                
    test_samples = len(test_dataset)


    print("Test images:", len(test_dataset))
    print("_TEST STATS__")
    test_counters = collections.Counter(test_labels)
    print(test_counters)


    preds = []
    bar = ChargingBar("Predicting ", max=(len(test_dl)))
    rows = []
    for index, (images, labels, images_paths, additional_images, additional_images_paths, ssim) in enumerate(test_dl):
        
        with torch.no_grad():
                
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(opt.gpu_id)
            labels = labels.unsqueeze(1).float()
            y_pred = model(images)
            
            if opt.model == 3:
                y_pred = y_pred.logits

            y_pred = y_pred.cpu()
            images = images.cpu()
            y_pred = [np.asarray(torch.sigmoid(pred).detach().numpy()) for pred in y_pred]

            additional_images = np.transpose(additional_images, (0, 3, 1, 2))
            additional_images = additional_images.to(opt.gpu_id)
            additional_y_pred = model(additional_images)
            
            if opt.model == 3:
                additional_y_pred = additional_y_pred.logits

            additional_y_pred = additional_y_pred.cpu()
            additional_images = additional_images.cpu()
            
            additional_y_pred = [np.asarray(torch.sigmoid(pred).detach().numpy()) for pred in additional_y_pred]
            delta_error = (labels[0] - y_pred[0]) - (labels[0] - additional_y_pred[0])
            
            
            rows.append({"SSIM": float(ssim), "Delta Error": float(delta_error)})

            bar.next()

    df = pd.DataFrame(rows, columns=["SSIM", "Delta Error"])
    print(df.dtypes)
    print(df["SSIM"].corr(df["Delta Error"]))