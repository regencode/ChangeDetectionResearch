import torch
import einops as ein

def calculate_metrics(y_true, y_pred, is_binary=True):
    '''
    Calculate and return accuracy, precision, recall, F1, IoU
    '''
    assert y_true.shape == y_pred.shape, f"y_true shape={y_true.shape} != y_pred shape{y_pred.shape}"
    if is_binary:
        TP = torch.sum((y_true == 1) & (y_pred == 1))
        TN = torch.sum((y_true == 0) & (y_pred == 0))
        FP = torch.sum((y_true == 0) & (y_pred == 1))
        FN = torch.sum((y_true == 1) & (y_pred == 0))
    else:
        print("Not implemented")
        return -1
    accuracy = (TP + TN) / (TP + TN + FP + FN) # proportion of correct predictions
    precision = TP / (TP + FP) # how correct given a prediction of a class
    recall = TP / (TP + FN) # how much of a class is accurately "recalled"
    F1 = (2 * precision * recall) / (precision + recall) # harmonic mean of precision and recall (???)

    IoU = TP / (TP + FP + FN) # how much overlap are there between prediction and label

    return accuracy.item(), precision.item(), recall.item(), F1.item(), IoU.item()


def get_confusion_matrix(y_true, y_pred, is_binary=True):
    '''
    Calculate and return accuracy, precision, recall, F1, IoU
    '''
    assert y_true.shape == y_pred.shape, f"y_true shape={y_true.shape} != y_pred shape{y_pred.shape}"
    if is_binary:
        TP = torch.sum((y_true == 1) & (y_pred == 1))
        TN = torch.sum((y_true == 0) & (y_pred == 0))
        FP = torch.sum((y_true == 0) & (y_pred == 1))
        FN = torch.sum((y_true == 1) & (y_pred == 0))
    else:
        print("Not implemented")
        return -1
    print(TP, TN, FP, FN)
    conf_array = torch.tensor([TP, FN, FP, TN])
    conf_mat = conf_array.view((2, 2)).numpy()
    return conf_mat

if __name__ == "__main__":
    N, W, H = 2, 20, 20
    true = torch.randint(0, 2, (N, W, H))
    pred = torch.randint(0, 2, (N, W, H))
    print(get_confusion_matrix(true, pred))



