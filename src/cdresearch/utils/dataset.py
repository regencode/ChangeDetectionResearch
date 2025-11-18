import torch
from torch.utils.data import Dataset

class InMemoryDataset(Dataset):
    def __init__(self, x1, x2, y, pair_transforms=None, return_y_image=False):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.pair_transforms = pair_transforms
        self.return_y_image = return_y_image

    def __getitem__(self, i: int):
        X_paths = (self.x1[i], self.x2[i])
        y_path = self.y[i]

        # open from filepath to tensor
        tensors: tuple[torch.Tensor, ...] = tuple(torch.from_numpy(cv.imread(path)).permute(2, 0, 1) for path in (*X_paths, y_path))
        x1, x2, y = tensors
        C, W, H = y.shape

        if self.pair_transforms:
            x1, x2, y = self.pair_transforms(x1, x2, y)

        y_label = torch.zeros((W, H), dtype=torch.long)
        y_label[:, :] = (y[0, :, :] == 255)

        X = torch.stack((x1, x2))

        return (X, y) if self.return_y_image else (X, y_label)

    def __len__(self) -> int :
        return len(self.y)
