import argparse
import copy
import warnings
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, loss_fn, lr_scheduler, datasets, num_epochs, batch_size):
    log_template = (
        "train_loss: {:.4f} | train_acc: {:.4f} | val_loss: {:.4f} | val_acc: {:.4f}"
    )
    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataloaders = {
        "train": DataLoader(
            datasets["train"], batch_size, shuffle=True, drop_last=True
        ),
        "val": DataLoader(datasets["val"], batch_size, shuffle=True, drop_last=True),
    }
    for epoch in range(num_epochs):
        epoch_data = {"train": {}, "val": {}}
        tqdm.write(f"Epoch: {epoch+1:03d}/{num_epochs:03d}")
        for mode in ("train", "val"):
            if mode == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for X, y in tqdm(dataloaders[mode], desc=mode, leave=False, unit="batch"):
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(mode == "train"):
                    outputs = model(X)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, y)

                    if mode == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * X.size(0)
                running_corrects += torch.sum(preds == y.data)

            if mode == "train":
                lr_scheduler.step()

            epoch_loss = running_loss / len(datasets[mode])
            epoch_acc = running_corrects.double() / len(datasets[mode])

            epoch_data[mode]["loss"] = epoch_loss
            epoch_data[mode]["acc"] = epoch_acc

            if mode == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        tqdm.write(
            log_template.format(
                epoch_data["train"]["loss"],
                epoch_data["train"]["acc"],
                epoch_data["val"]["loss"],
                epoch_data["val"]["acc"],
            )
        )

    model.load_state_dict(best_model_wts)
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--val", type=str, required=True)
parser.add_argument("-b", "--batch-size", type=int, required=True)
parser.add_argument("-e", "--epochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)

args = parser.parse_args()
args = vars(args)

datasets = {
    "train": datasets.ImageFolder(
        args.get("train"),
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    ),
    "val": datasets.ImageFolder(
        args.get("val"),
        transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    ),
}

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, len(datasets["train"].classes))
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), args.get("lr"))
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.2)
loss_fn = torch.nn.CrossEntropyLoss()

model = train(
    model,
    optimizer,
    loss_fn,
    lr_scheduler,
    datasets,
    args.get("epochs"),
    args.get("batch_size"),
)
model = model.to("cpu")

torch.save(model.state_dict(), "model.pt")
