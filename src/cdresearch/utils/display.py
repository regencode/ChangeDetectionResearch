import torch
import numpy as np
import math
import matplotlib.pyplot as plt


def display_images(image_dict: dict[str, str | np.ndarray | torch.Tensor], 
                   rows: int | None = None, cols: int = 4, 
                   figsize: tuple[int, int] = (10, 10),
                   print_dict: bool = False): 
    '''
    Display images when passed a dict {title: image}
    title: string
    image: string | np.ndarray | torch.Tensor
    Default: 
        - images are displayed in 4 columns
        - figsize (5, 5)
    '''
    num_images = len(image_dict.keys())
    print(f"image_dict has {num_images} images")
    if rows is None:
        rows = math.ceil(float(num_images) / cols)
    print(f"using cols: {cols} and rows: {rows}")
    print(f"using figsize: {figsize}")
    print("image dict:", image_dict, flush=True) if print_dict else None
    plt.figure(figsize=figsize)
    for i, (title, image) in enumerate(image_dict.items()):
        plt.subplot(rows, cols, i+1)
        try:
            if isinstance(image, str):
                # handle path
                image = plt.imread(image)
            elif isinstance(image, torch.Tensor): #(C, W, H)
                image = image.cpu()

        except Exception as e:
            print("Exception:", e)

        plt.imshow(image)
        plt.title(title)
    plt.show()

def display_during_inference(X_batch, y_binary, outputs_binary):
    '''
    Display first tensor in batch only.
    '''
#    X_batch = X_batch.cpu()
#    y_batch = y_batch.cpu()
#    y_binary = y_binary.cpu()
#    outputs_1 = outputs_1.cpu()
#    outputs_2 = outputs_2.cpu()
#    outputs_binary = outputs_binary.cpu()
    N, _, C, W, H = X_batch.shape
    # cannot use tensor.view because X is likely to have been permuted earlier,
    # causing non-contiguous memory positions.

    X_batch = X_batch.permute(0, 1, 3, 4, 2) # (N, 2, W, H, C)
    rand_idx = np.random.randint(0, N)
    x1, x2 = X_batch[rand_idx] # (W, H, C)
    # (W, H, C)
    pred_binary = torch.argmax(outputs_binary, dim=1)[rand_idx]
    y = y_binary[rand_idx]

    args = {
        "rows": 2,
        "cols": 2,
        "figsize" : (12, 12)
    }
    display_images({
        "X_before" : x1.cpu(),
        "X_after" : x2.cpu(),
        "y_binary_changes": y.cpu(),
        "pred_binary_changes": pred_binary.cpu()
        }, **args
    )

