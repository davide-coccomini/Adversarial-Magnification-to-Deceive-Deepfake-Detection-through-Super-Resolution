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
from sklearn.metrics import accuracy_score
from progress.bar import ChargingBar
from utils import check_correct, unix_time_millis
from timm.scheduler.cosine_lr import CosineLRScheduler
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.metrics import f1_score

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
    parser.add_argument('--data_path', default='../deep_fakes/datasets/processed/crops_ff_minimized', type=str,
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
    parser.add_argument('--forgery_method', type=int, default=1, 
                        help="Forgery method used for training")
    parser.add_argument('--save_errors', type=bool, default=True, 
                        help="Save errors in directory?")
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
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

    
    model.load_state_dict(torch.load(opt.model_path))

    model = model.to(opt.gpu_id)
    model.eval()

    
    loss_fn = torch.nn.BCEWithLogitsLoss()


    if opt.dataset == 1:
        images_paths = []
        test_labels = []
        df_test = pd.read_csv(opt.list_file, names=["video", "label"], sep=" ")
        
        for index, row in df_test.iterrows():
            video_path = os.path.join(opt.data_path, row["video"])
            image_name = os.listdir(video_path)[0]
            image_path = os.path.join(video_path, image_name)
            images_paths.append(image_path)
            test_labels.append(row["label"])

    
    if opt.max_images > -1:
        images_paths = images_paths[:opt.max_images]
        test_labels = test_labels[:opt.max_images]


    test_dataset = DeepFakesDataset(images_paths, test_labels, mode='val')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config['test']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)

                                
    test_samples = len(test_dataset)


    print("Test images:", len(test_dataset))
    print("_TEST STATS__")
    test_counters = collections.Counter(test_labels)
    print(test_counters)


    test_correct = 0
    total_test_loss = 0
    test_correct = 0
    test_positive = 0
    test_negative = 0
    test_counter = 0

    preds = []
    bar = ChargingBar("Predicting ", max=(len(test_dl)))
    for index, (images, labels) in enumerate(test_dl):
        
        with torch.no_grad():
                
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(opt.gpu_id)
            labels = labels.unsqueeze(1).float()
            y_pred = model(images)
            y_pred = y_pred.cpu()
            images = images.cpu()
            
            test_loss = loss_fn(y_pred, labels)
            total_test_loss += round(test_loss.item(), 2)
            corrects, positive_class, negative_class, _ = check_correct(y_pred, labels)  
            
            y_pred = y_pred.numpy()[0]
            test_correct += corrects
            test_positive += positive_class
            test_counter += 1
            test_negative += negative_class
            preds.extend(y_pred)
            bar.next()


    
    total_test_loss /= test_counter
    test_correct /= test_samples
    preds = [torch.sigmoid(torch.tensor(pred)) for pred in preds]
    fpr, tpr, th = metrics.roc_curve(test_labels, [pred.item() for pred in preds])
    auc = metrics.auc(fpr, tpr)
    rounded_preds = [round(pred.item()) for pred in preds]
    f1 = f1_score(test_labels, rounded_preds)
    accuracy = accuracy_score(test_labels, rounded_preds)

    print(str(opt.model_path) + " test loss:" + str(total_test_loss) + " f1 score: " + str(f1) + " test accuracy:" + str(test_correct) + " test_0s:" + str(test_negative) + "/" + str(test_counters[0]) + " test_1s:" + str(test_positive) + "/" + str(test_counters[1]) + " AUC " + str(auc))
    