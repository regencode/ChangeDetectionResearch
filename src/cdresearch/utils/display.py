import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import einops as ein


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

def display_during_inference(X_batch : torch.Tensor, y_binary : torch.Tensor, outputs_binary : torch.Tensor):
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

    X_batch = X_batch.permute(0, 1, 3, 4, 2) # (N, 2, H, W, C)
    x1, x2 = X_batch[0] # (H, W, C)
    # (H, W, C)
    pred_binary = torch.argmax(outputs_binary, dim=1)[0]
    y = y_binary[0]
    new_pred_display = torch.zeros((W, H, C), dtype=torch.long)
    new_label_display = torch.zeros((W, H, C), dtype=torch.long)

    # red: overpredict
    # blue: underpredict
    tp = (pred_binary == 1) & (y == 1)
    tn = (pred_binary == 0) & (y == 0)
    fp = (pred_binary == 1) & (y == 0)
    fn = (pred_binary == 0) & (y == 1)

    new_pred_display[tp] = torch.tensor((255, 255, 255))
    new_pred_display[fp] = torch.tensor((255, 0, 0))
    new_pred_display[fn] = torch.tensor((0, 0, 255))

    new_label_display[y == 1] = torch.tensor((255, 255, 255))
    
    args = {
        "rows": 2,
        "cols": 2,
        "figsize" : (12, 12)
    }
    display_images({
        "X_before" : x1.cpu(),
        "X_after" : x2.cpu(),
        "y_binary_changes": new_label_display.cpu(),
        "pred_binary_changes": new_pred_display.cpu()
        }, **args
    )

