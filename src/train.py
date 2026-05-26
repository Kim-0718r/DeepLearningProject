from pathlib import Path

import torch

try:
    from .config import Config
except ImportError:
    from config import Config


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }
    torch.save(checkpoint, save_path)


def build_optimizer(model, model_name, cfg=Config):
    learning_rate = cfg.LEARNING_RATES.get(model_name)

    if learning_rate is None:
        raise ValueError(f"Learning rate is not defined for model_name: {model_name}")

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]

    return torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=cfg.WEIGHT_DECAY,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=Config.EPOCHS,
    checkpoint_path=None,
):
    model = model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"train_loss: {train_loss:.4f} "
            f"train_acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} "
            f"val_acc: {val_acc:.4f}"
        )

        if checkpoint_path is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                save_path=checkpoint_path,
            )

    return history


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
