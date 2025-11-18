import os, glob
import shutil
import zipfile
import torch
import cv2 as cv
from torchvision.utils import save_image
from .base_dataset import BaseDataset

# Preprocessing: Patchify
class Patchify:
    def __init__(self, ph=256, pw=256):
        self.ph = ph
        self.pw = pw
    def __call__(self, x):
        '''
        from image of shape (C, H, W),
        return tensor of shape (N, C, new_H, new_W)
        where N is the number of patches in the input image
        '''
        C, H, W = x.shape

        ph = self.ph
        pw = self.pw
        x = x.unsqueeze(0)

        x = x.unfold(2, ph, ph).unfold(3, pw, pw)

        assert x.shape == (1, C, H//ph, W//pw, ph, pw), x.shape
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, ph, pw)
        return x

def load_levir(drive_path, patchify=False, patch_size=(256, 256)):
    DATA_SOURCE = drive_path
    DATA_DEST = "./LEVIR_CD"
    DATA_PATCH_FOLDER = "./LEVIR_CD_patched/"

    if os.path.exists(DATA_DEST):
        print("Data patch folder already exists! Skipping loading and unzipping data...")
        return

    os.makedirs(DATA_DEST)
    data_splits = glob.glob(DATA_SOURCE + "*.zip")
    for split in data_splits:
        dest = shutil.copy(split, DATA_DEST)
        os.makedirs(DATA_DEST + "/" + dest.split("/")[-1][:-4])
        with zipfile.ZipFile(split, 'r') as z:
            z.extractall(DATA_DEST + "/" + dest.split("/")[-1][:-4])
        print("Data load and unzip complete")

    if not patchify:
        print("Skip patchify data (patchify == False)")
        return

    patcher = Patchify(*patch_size)
    if os.path.exists(DATA_PATCH_FOLDER):
        print("Data patch folder already exists! Skipping patchify...")
        return
    os.makedirs(DATA_PATCH_FOLDER)
    os.makedirs(DATA_PATCH_FOLDER + "train/")
    os.makedirs(DATA_PATCH_FOLDER + "test/")
    os.makedirs(DATA_PATCH_FOLDER + "val/")
    for split in os.listdir(DATA_PATCH_FOLDER):
        split_path = DATA_PATCH_FOLDER + "/" + split
        os.makedirs(split_path + "/" + "A/")
        os.makedirs(split_path + "/" + "B/")
        os.makedirs(split_path + "/" + "label/")
        for subsplit in os.listdir(split_path):
            subsplit_path = split_path + "/" + subsplit

            for image_path in glob.glob(DATA_DEST + "/" + split + "/" + subsplit + "/*.png"):
                img = torch.from_numpy(cv.imread(image_path)).permute(2, 0, 1)
                # patchify
                img = patcher(img)
                for i, patch in enumerate(img):
                    save_image(patch.float()/255.0, subsplit_path + "/" + image_path.split("/")[-1][:-4] + f"_{i}.png")

    print("Data patchify complete")

class LEVIR_CD_Dataset(BaseDataset):
    def __init__(self, root="./LEVIR_CD", split="train", pair_transforms=None, return_y_image=False):

        x1_dir = f"{root}/{split}/A/"
        x2_dir = f"{root}/{split}/B/"
        mask_dir = f"{root}/{split}/label/"

        x1_paths = glob.glob(f"{x1_dir}/*.png")
        x2_paths = glob.glob(f"{x2_dir}/*.png")
        mask_paths = glob.glob(f"{mask_dir}/*.png")

        super().__init__(x1_paths, x2_paths, mask_paths, pair_transforms, return_y_image)


