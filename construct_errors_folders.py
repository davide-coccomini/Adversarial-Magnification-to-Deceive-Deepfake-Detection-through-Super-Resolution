import os
import cv2
import shutil
from albumentations import Compose, PadIfNeeded
from transforms.albu import IsotropicResize
RESULTS_PATHS = "outputs/test/all_metrics_tests_scale2_pristine"
BASE_IMAGE_PATH = "../deep_fakes/datasets/processed/crops_ff_minimized10"
txt_files = sorted(os.listdir(RESULTS_PATHS))
txt_files = [txt_file for txt_file in txt_files if "_errors" in txt_file and "magnified" not in txt_file ]

def create_val_transforms(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def resize_image(image, dim):
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized



for txt_file in txt_files:
    model_name = txt_file.split("_")[0]
    method = txt_file.split("_")[-3]
    without_sr_errors_f = open(os.path.join(RESULTS_PATHS, txt_file), "r")
    without_sr_errors = without_sr_errors_f.read()
    without_sr_errors = without_sr_errors.split("\n")
    without_sr_root = without_sr_errors[0].split("minimized10/")
    without_sr_errors = [line.split("minimized10/")[-1] for line in without_sr_errors]
    without_sr_errors_f.close()

    with_sr_errors_f = open(os.path.join(RESULTS_PATHS, txt_file).replace("extended", "magnified_extended_v3_cross"))
    with_sr_errors = with_sr_errors_f.read()
    with_sr_errors = with_sr_errors.split("\n")
    with_sr_errors = [line.split("scale2/")[-1] for line in with_sr_errors]
    
    with_sr_errors_f.close()

    
    if len(without_sr_errors) > len(with_sr_errors):
        diff = list(set(without_sr_errors) - set(with_sr_errors))
    else:
        diff = list(set(with_sr_errors) - set(without_sr_errors))
    errors_path = os.path.join(RESULTS_PATHS, "errors")
    method_path = os.path.join(errors_path, method)
    model_path = os.path.join(method_path, model_name)
    os.makedirs(model_path, exist_ok=True)
   
    for error in diff:
        if error == "":
            continue
        error_src = os.path.join(BASE_IMAGE_PATH,  error)
        error_dst = os.path.join(model_path,  error.replace("/", "_"))
        error_sr_src = os.path.join(BASE_IMAGE_PATH.replace("crops_ff_minimized10", "crops_ff_minimized10_magnified_v3_scale2"),  error)
        error_sr_dst = os.path.join(model_path,  error.replace("/", "_").replace(".png", "_sr.png"))
        error_src = error_src.replace("crops_ff_minimized10", "crops_ff")
        image = cv2.imread(error_src)
        sr_image = cv2.imread(error_sr_src)
        transform = create_val_transforms(224)
        cv2.imwrite(error_dst, transform(image=image)["image"])        
        cv2.imwrite(error_sr_dst, transform(image=sr_image)["image"])

        