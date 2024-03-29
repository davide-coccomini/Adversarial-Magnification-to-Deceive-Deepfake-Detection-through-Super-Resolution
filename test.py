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
    parser.add_argument('--save_table', type=str, default=None,
                        help="Save table in directory with results")
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--forgery_method', type=str, default='',
                        help="")
    parser.add_argument('--explanation_path', type=str, default='',
                        help="")
    parser.add_argument('--save_errors', type=str, default="", 
                        help="")
    parser.add_argument('--save_rates', type=str, default="", 
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
        if opt.explanation_path != '':
            explanation_layers = [model.layers[-1].blocks[-1].norm1]
    elif opt.model == 1: 
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(2048, config['model']['num-classes'])
        if opt.explanation_path != '':
            explanation_layers = [model.layer4[-1]]
    elif opt.model == 2:
        model = timm.create_model('xception', pretrained=True, num_classes = config['model']['num-classes'])
        if opt.explanation_path != '':
            explanation_layers = [model.block12[-1]]
    elif opt.model == 3:
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=config['model']['num-classes'])
        if opt.explanation_path != '':
            explanation_layers = [model.blocks[-1].norm1]

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

    
    if opt.explanation_path != "":
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)


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


    test_dataset = DeepFakesDataset(images_paths, test_labels, mode='val')
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


    test_correct = 0
    total_test_loss = 0
    test_correct = 0
    test_positive = 0
    test_negative = 0
    test_counter = 0

    preds = []
    bar = ChargingBar("Predicting ", max=(len(test_dl)))
    if opt.save_errors != "":
        errors = ""

    for index, (images, labels, images_paths) in enumerate(test_dl):
        
        with torch.no_grad():
                
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(opt.gpu_id)
            labels = labels.unsqueeze(1).float()
            y_pred = model(images)
            
            if opt.model == 3:
                y_pred = y_pred.logits

            y_pred = y_pred.cpu()
            images = images.cpu()
            
            test_loss = loss_fn(y_pred, labels)
            total_test_loss += round(test_loss.item(), 2)
            corrects, positive_class, negative_class, sig_preds = check_correct(y_pred, labels)  
            if opt.save_errors != "":
                if corrects == 0 and labels[0] == 1:
                    errors += images_paths[0] + "\n"
            
            y_pred = y_pred.numpy()[0]
            test_correct += corrects
            test_positive += positive_class
            test_counter += 1
            test_negative += negative_class
            preds.extend(y_pred)


            bar.next()

        if opt.save_errors:
            fe = open(opt.save_errors, "w+")
            fe.write(errors)
            fe.close()

    
    total_test_loss /= test_counter
    test_correct /= test_samples
    preds = [torch.sigmoid(torch.tensor(pred)) for pred in preds]
    fpr, tpr, th = metrics.roc_curve(test_labels, [pred.item() for pred in preds])
    auc = round(metrics.auc(fpr, tpr), 3) * 100
    auc_fpr = fpr
    auc_tpr = tpr
    rounded_preds = [float(round(pred.item())) for pred in preds]
    _tn, _fp, _fn, _tp = confusion_matrix(test_labels, rounded_preds).ravel()
    _tpr = _tp / (_tp + _fn)
    _fpr = _fp / (_tn + _fp)
    _fnr = _fn / (_tp + _fn)
    _tnr = _tn / (_tn + _fp)
    fpr = round(_fpr, 3) * 100
    tpr = round(_tpr, 3) * 100
    fnr = round(_fnr, 3) * 100
    tnr = round(_tnr, 3) * 100
    test_correct = round(test_correct, 3) * 100
    f1 = round(f1_score(test_labels, rounded_preds), 3) * 100
    accuracy = round(accuracy_score(test_labels, rounded_preds), 3) * 100
    precision = round(precision_score(test_labels, rounded_preds), 3) * 100
    print("UNROUNDED PRECISION", precision_score(test_labels, rounded_preds))
    recall = round(recall_score(test_labels, rounded_preds), 3) * 100

    print(str(opt.model_path) + " test loss:" + str(total_test_loss) + " FPR: " + str(fpr) +  " FNR: "+ str(fnr) +" TPR: " + str(tpr) + " TNR: " + str(tnr) + "\nf1 score: " + str(f1) + " test accuracy: " + str(test_correct) + " test_precision: " + str(precision) + " test recall: "+ str(recall) + " test_0s:" + str(test_negative) + "/" + str(test_counters[0]) + " test_1s:" + str(test_positive) + "/" + str(test_counters[1]) + " AUC " + str(auc))


    if "magnified" in opt.fake_data_path:
        sr = "$\\checkmark$"
        sr_bool = True
    else:
        sr = "$\\times$"
        sr_bool = False
    if opt.save_table is not None:
        f = open(opt.save_table, "a+")
        if os.stat(opt.save_table).st_size == 0:
            f.write("Model & Forgery Method & SR & FNR & FPR & Recall & Precision & AUC & Accuracy \\\\")

        f.write("\n" + opt.model_name + " & " + opt.forgery_method + " & " + sr + " & " + str(fnr) + " & " + str(fpr) + " & " + str(recall) + " & " + str(precision) + " & " + str(auc) + " & " + str(test_correct) + "\\\\")
        f.close()

    if len(opt.save_rates) > 1:
        f = open(opt.save_rates, "a+")
        if os.stat(opt.save_rates).st_size == 0:
            f.write("Model,Method,SR,FPR,TPR,TH\n")

        f.write(opt.model_name + "," + opt.forgery_method + "," + str(sr_bool) + "," + convert_list_to_string(auc_fpr) + "," + convert_list_to_string(auc_tpr) + "," + convert_list_to_string(th) + "\n")
        f.close()
