import os, glob
import shutil
import zipfile
import torch
import cv2 as cv
from torchvision.utils import save_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from .base_dataset import BaseDataset, Patchify

def load_whu(drive_path, patchify=False, patch_size=(256, 256)):
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_SOURCE = drive_path
    DATA_DEST = f"{MODULE_DIR}/WHU_CD"
    DATA_PATCH_FOLDER = f"{MODULE_DIR}/WHU_CD_PATCHED/"

    if os.path.exists(DATA_DEST):
        print("Data unzip dest folder already exists! Skipping loading data...")
    else: 
        os.makedirs(DATA_DEST)
        dest = shutil.copy(DATA_SOURCE, DATA_DEST)
        with zipfile.ZipFile(dest, 'r') as z:
            z.extractall(DATA_DEST + "/" + dest.split("/")[-1][:-4])
        print("Data load and unzip complete")

    if not patchify:
        print("Skip patchify data (patchify == False)")
        return

    patcher = Patchify(*patch_size)
    #if os.path.exists(DATA_PATCH_FOLDER):
    #    print("Data patch folder already exists! Skipping patchify...")
    #    return
    os.makedirs(DATA_PATCH_FOLDER)
    os.makedirs(DATA_PATCH_FOLDER + "train/")
    os.makedirs(DATA_PATCH_FOLDER + "test/")
    for split in os.listdir(DATA_PATCH_FOLDER):
        split_path = os.path.join(DATA_PATCH_FOLDER, split)
        os.makedirs(split_path + "/" + "A/")
        os.makedirs(split_path + "/" + "B/")
        os.makedirs(split_path + "/" + "label/")
        image_path_prefix = os.path.join(DATA_DEST, f"Building change detection dataset_add/1. The two-period image data/")
        image_dict = {
            "A": os.path.join(image_path_prefix, f"2012/whole_image/{split}/image/2012_{split}.tif"),
            "B": os.path.join(image_path_prefix, f"2016/whole_image/{split}/image/2016_{split}.tif"),
            "label": os.path.join(image_path_prefix, f"change_label/{split}/change_label.tif")
        }
        print(image_dict)
        for subsplit in os.listdir(split_path):
            subsplit_path =  os.path.join(split_path, subsplit)
            image_path = image_dict[subsplit]
            img = torch.from_numpy(cv.imread(image_path)).permute(2, 0, 1)
            # patchify
            img = patcher(img)
            for i, patch in enumerate(img):
                save_image(patch.float()/255.0, os.path.join(subsplit_path, image_path.split("/")[-1][:-4] + f"_{i}.png"))

    print("Data patchify complete")

class WHU_CD_Dataset(BaseDataset):
    def __init__(self, root="./WHU_CD", split="train", pair_transforms=None, return_y_image=False):
        '''
        assume data is already patchified.
        splits: train, test
        '''

        x1_dir = f"{root}/{split}/A/"
        x2_dir = f"{root}/{split}/B/"
        mask_dir = f"{root}/{split}/label/"

        x1_paths = glob.glob(f"{x1_dir}/*.png")
        x2_paths = glob.glob(f"{x2_dir}/*.png")
        mask_paths = glob.glob(f"{mask_dir}/*.png")

        super().__init__(x1_paths, x2_paths, mask_paths, pair_transforms, return_y_image)
