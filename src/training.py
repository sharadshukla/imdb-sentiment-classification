"""
Training utilities for neural sentiment models.

Includes:
- generic neural-network training loop
- learning-rate warmup and decay scheduler
"""

import torch.nn as nn
import torch.optim as optim


def train_model(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    n_epochs=10,
    scheduler=None
):
    """
    Train a neural sentiment model.

    Parameters
    ----------
    model : torch.nn.Module
        Neural model to train.

    dataloader : torch.utils.data.DataLoader
        Training dataloader.

    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.

    criterion : torch.nn.Module
        Loss function, e.g. BCEWithLogitsLoss.

    device : torch.device
        CPU or GPU device.

    n_epochs : int
        Number of training epochs.

    scheduler : optional
        Learning-rate scheduler.

    Returns
    -------
    epoch_losses : list[float]
        Average training loss for each epoch.
    """

    model.to(device)

    epoch_losses = []

    for epoch in range(1, n_epochs + 1):

        model.train()

        total_loss = 0.0

        for X, lengths, y in dataloader:
            
            # Move model inputs and labels to CPU/GPU
            X = X.to(device)
            y = y.to(device)

            # Sequence lengths remain on CPU for packed recurrent sequences
            lengths = lengths.cpu()

            # Clear gradients from previous step
            optimizer.zero_grad()

            # Forward pass
            logits = model(X, lengths).squeeze(1)

            # Compute binary classification loss
            loss = criterion(logits, y)

            # Backpropagation
            loss.backward()

            # Prevent exploding gradients
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

            # Update learning rate after each epoch
        if scheduler:
            scheduler.step()

        avg_loss = total_loss / len(dataloader)

        epoch_losses.append(avg_loss)

        print(
            f"Epoch {epoch:>2}/{n_epochs} | "
            f"Loss: {avg_loss:.4f}"
        )

    return epoch_losses

def make_scheduler(
    optimizer,
    n_epochs,
    warmup_epochs
):
    """
    Create a learning-rate warmup and linear-decay scheduler.

    The learning rate increases gradually during the warmup period
    and then decreases linearly for the remaining epochs.

    This was used for the self-attention model in the original
    experiment to make early training more stable.
    """

    def lr_lambda(epoch):

        # Warmup phase
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        # Linear decay phase
        return max(
            0.0,
            (n_epochs - epoch)
            / (n_epochs - warmup_epochs)
        )

    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )