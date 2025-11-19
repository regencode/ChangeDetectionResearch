from typing import Optional
import torch
from .metrics import calculate_metrics
from .display import display_during_inference

def y_to_binary(y):
    y_binary = torch.zeros_like(y)
    y_binary[y != 0] = 1
    return y_binary

def batch_inference(model, X_batch, y_batch, binary_loss_fn, optimizer, train=False, device="cuda", **kwargs):
        # concat X along channel
        X_batch = X_batch.float() / 255.0

        # access each X and y member in pair
        X_batch = X_batch.permute(1, 0, 2, 3, 4)
        # Load data and move to device

        X_batch = X_batch.to(device)
        y_binary = y_batch.to(device) # (N, P, W, H)

        # Forward pass
        outputs_binary = model(X_batch[0], X_batch[1])
        if train:
            loss = binary_loss_fn(outputs_binary, y_binary)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            loss = binary_loss_fn(outputs_binary, y_binary)
        return loss.item(), y_binary, outputs_binary


def train_one_epoch(model, dataloader, optimizer, loss_fn, device="cuda"):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    batch_losses = []
    for i, (X_batch, y_batch) in enumerate(dataloader):
        loss, *_ = batch_inference(model, X_batch, y_batch, loss_fn, optimizer, train=True, device=device)
        running_loss += loss
        print(f"\r[Train {i+1}/{len(dataloader)}]Batch Loss: {loss} | Running Loss: {running_loss/(i+1)}", end="")
        batch_losses.append(loss)

    return running_loss / len(dataloader), batch_losses

def val_one_epoch(model, dataloader, optimizer, loss_fn, device, display_inference=False):
    model.eval()  # Set the model to training mode
    running_loss = 0.0
    batch_losses = []
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(dataloader):
            loss, y_binary, output_binary = batch_inference(model, X_batch, y_batch, loss_fn, optimizer, train=False, device=device)
            running_loss += loss

            print(f"\r[Val {i+1}/{len(dataloader)}]Batch Loss: {loss} | Running Loss: {running_loss/(i+1)}", end="")
            batch_losses.append(loss)

            if i % 100 == 0 and display_inference:
                display_during_inference(X_batch, y_binary, output_binary)
    return running_loss / len(dataloader), batch_losses

def test_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.eval()  # Set the model to training mode
    running_loss = 0.0
    batch_losses = []

    X_batch_stack = None
    y_binary_stack = None
    output_binary_stack = None

    stacks = [X_batch_stack, y_binary_stack, output_binary_stack]

    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(dataloader):
            loss, y_binary, output_binary = batch_inference(model, X_batch, y_batch, loss_fn, optimizer, train=False, device=device)
            running_loss += loss
            print(f"\r[Test{i+1}/{len(dataloader)}]Batch Loss: {loss} | Running Loss: {running_loss/(i+1)}", end="")
            batch_losses.append(loss)
            batches = [X_batch, y_binary, output_binary]

            for j, (stack, batch) in enumerate(zip(stacks, batches)):
                if stacks[j] is None:
                    stacks[j] = batch.detach().cpu()
                else:
                    stacks[j] = torch.cat((stack, batch.detach().cpu()), dim=0)
            if i % 10 == 0:
                display_during_inference(X_batch, y_binary, output_binary)

    X_batch_stack, y_binary_stack, output_binary_stack = stacks
    output_binary_stack = torch.argmax(output_binary_stack, dim=1)
    accuracy, precision, recall, F1, IoU = calculate_metrics(y_binary_stack, output_binary_stack)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "IoU": IoU
    }
    print(metrics)
    return running_loss / len(dataloader), batch_losses, metrics

