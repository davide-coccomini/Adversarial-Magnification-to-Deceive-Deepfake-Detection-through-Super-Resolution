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
import timm
from progress.bar import ChargingBar
from utils import check_correct, unix_time_millis
from timm.scheduler.cosine_lr import CosineLRScheduler
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.metrics import f1_score
from transformers import ViTForImageClassification, ViTConfig

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)  
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--data_path', default='../deep_fakes/datasets/processed/crops_ff_minimized10', type=str,
                        help='Images directory')
    parser.add_argument('--list_file', default="../deep_fakes/datasets/training_videos.csv", type=str,
                        help='Images List json file path)')
    parser.add_argument('--val_data_path', default='../deep_fakes/datasets/processed/crops_ff_minimized10', type=str,
                        help='Images directory')
    parser.add_argument('--val_list_file', default="../deep_fakes/datasets/validation_videos.csv", type=str,
                        help='Images List json file path')
    parser.add_argument('--dataset', default=1, type=int,
                        help='Dataset to be processed (0: Openforensics; 1: FF++)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Model name.')
    parser.add_argument('--model_path', type=str, default='models',
                        help='Path to save checkpoints.')
    parser.add_argument('--max_images', type=int, default=-1, 
                        help="Maximum number of images to use for training (default: all).")
    parser.add_argument('--config', type=str, default='',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--forgery_method', type=str, default='',
                        help="")
    parser.add_argument('--model', type=int, default=0, 
                        help="Which model architecture version to be trained (0: Swin, 1: Resnet, 2: EfficientNet")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    parser.add_argument('--show_stats', type=bool, default=True, 
                        help="Show stats")
    parser.add_argument('--logger_name', default='runs/train',
                        help='Path to save the model and Tensorboard log.')
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
    elif opt.model == 2:
        model = timm.create_model('xception', pretrained=True, num_classes = config['model']['num-classes'])
    elif opt.model == 3:
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', ignore_mismatched_sizes=True, num_labels=config['model']['num-classes'])

    model = model.to(opt.gpu_id)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])



    # Data loading

    if opt.dataset == 0:
        f = open(opt.list_file)
        annotations_json = dict(json.load(f))["annotations"]
        images_paths = []
        labels = []
        for item in annotations_json:
            item_path = os.path.join(opt.data_path, str(item["image_id"]))
            label = item["category_id"]
            
            if os.path.exists(item_path):
                for image in os.listdir(item_path):
                    if "0" not in image:
                        continue
                    image_path = os.path.join(item_path, image)
                    images_paths.append(image_path)
                    labels.append(label)



        f = open(opt.val_list_file)
        annotations_json = dict(json.load(f))["annotations"]
        val_images_paths = []
        val_labels = []
        for item in annotations_json:
            item_path = os.path.join(opt.data_path, str(item["image_id"]))
            label = item["category_id"]
            
            if os.path.exists(item_path):
                for image in os.listdir(item_path):
                    if "0" not in image:
                        continue
                    image_path = os.path.join(item_path, image)
                    val_images_paths.append(image_path)
                    val_labels.append(label)
    elif opt.dataset == 1:
        images_paths = []
        labels = []
        df_train = pd.read_csv(opt.list_file, names=["video", "label"], sep=" ")
        
        for index, row in df_train.iterrows():
            video_path = os.path.join(opt.data_path, row["video"])
            if opt.forgery_method not in video_path and "Original" not in video_path:
                continue

            for image_name in os.listdir(video_path):
                image_path = os.path.join(video_path, image_name)
                images_paths.append(image_path)
                labels.append(row["label"])

        val_images_paths = []
        val_labels = []
        df_val = pd.read_csv(opt.val_list_file, names=["video", "label"], sep=" ")
        for index, row in df_val.iterrows():
            video_path = os.path.join(opt.data_path, row["video"])
            if opt.forgery_method not in video_path and "Original" not in video_path:
                continue
            for image_name in os.listdir(video_path):
                image_path = os.path.join(video_path, image_name)
                val_images_paths.append(image_path)
                val_labels.append(row["label"])


    if opt.max_images > -1:
        images_paths = images_paths[:opt.max_images]
        labels = labels[:opt.max_images]
        val_images_paths = val_images_paths[:opt.max_images]
        val_labels = val_labels[:opt.max_images]


    train_dataset = DeepFakesDataset(images_paths, labels)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)

    
    val_dataset = DeepFakesDataset(val_images_paths, val_labels, mode='val')
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['val_bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)


    # Print some useful statistics

    
    train_samples = len(train_dataset)
    validation_samples = len(val_dataset)


    print("Train images:", len(train_dataset), "Validation images:", len(val_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(labels)
    print(train_counters)
    
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(val_labels)
    print(val_counters)
    print("___________________")



    # Epoch Loop
    
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    num_steps = int(opt.num_epochs * len(train_dl))
    lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=config['training']['lr'] * 1e-2,
                cycle_limit=2,
                t_in_epochs=False,
        )


    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(0, opt.num_epochs + 1):
        model.train()
        if not_improved_loss == opt.patience:
            break

        # Init epoch variables
        counter = 0
        total_loss = 0
        total_val_loss = 0
        train_correct = 0
        positive = 0
        negative = 0
        train_batches = len(train_dl)
        val_batches = len(val_dl)
        total_batches = train_batches + val_batches

        bar = ChargingBar('EPOCH #' + str(t), max=(len(train_dl)+len(val_dl)))
        for index, (images, labels, _) in enumerate(train_dl):
            start_time = datetime.now()
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(opt.gpu_id)
            labels = labels.unsqueeze(1).float()
            

            y_pred = model(images)
            if opt.model == 3:
                y_pred = y_pred.logits

            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels)
            corrects, positive_class, negative_class, _ = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            counter += 1
            total_loss += round(loss.item(), 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lr_scheduler.step_update((t * (train_batches) + index))
            time_diff = unix_time_millis(datetime.now() - start_time)

            bar.next()
            if index%100 == 0:
                expected_time = str(datetime.fromtimestamp((time_diff)*(total_batches-index)/1000).strftime('%H:%M:%S.%f'))
                print("\nLoss: ", total_loss/counter, "Accuracy: ", train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive, "Expected Time:", expected_time)


        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        train_correct /= train_samples
        total_loss /= counter
        val_preds = []
        model.eval()

        
        for index, (images, labels, _) in enumerate(val_dl):
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.to(opt.gpu_id)
            labels = labels.unsqueeze(1).float()
            with torch.no_grad():
                val_pred = model(images)
                    
                if opt.model == 3:
                    val_pred = val_pred.logits
                val_pred = val_pred.cpu()
                val_loss = loss_fn(val_pred, labels)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class, val_pred = check_correct(val_pred, labels)
                val_correct += corrects
                val_positive += positive_class
                val_counter += 1
                val_negative += negative_class
                val_preds.extend(val_pred)
                bar.next()

        
        bar.finish()

            
        total_val_loss /= val_counter
        val_correct /= validation_samples

        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        if previous_loss > total_val_loss:
            os.makedirs(opt.model_path, exist_ok = True)
            torch.save(model.state_dict(), os.path.join(opt.model_path, opt.model_name + "_checkpoint" + str(t)))

        previous_loss = total_val_loss
         

        fpr, tpr, th = metrics.roc_curve(val_labels, [pred.item() for pred in val_preds])
        auc = metrics.auc(fpr, tpr)
        f1 = f1_score(val_labels, [round(pred.item()) for pred in val_preds])

        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(val_counters[0]) + " val_1s:" + str(val_positive) + "/" + str(val_counters[1]) + " val_auc: " + str(auc) + " val_f1: " + str(f1))

    