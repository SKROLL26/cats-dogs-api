import argparse
from server.cnn.model import CNN
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import torch
import pathlib
from PIL import Image
from tqdm import tqdm
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore")
torch.manual_seed(182)
random.seed(182)
np.random.seed(182)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CatsDogsDataset(Dataset):

    def __init__(self, files:list[pathlib.Path], labels):
        super().__init__()
        self.files = sorted(files)
        self.labels = labels
    

    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, i):
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img = Image.open(self.files[i])
        img.load()
        img = transform(img)
        label = self.labels[i]
        return img, label

def train(model, optimizer, lr_scheduler, num_epochs, batch_size, loss_fn, device, train_data, val_data):
    train_loader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, drop_last=True, shuffle=True)
    log_template = "Epoch: {:03d} | train_loss: {:.4f} | val_loss: {:.4f} | train_acc: {:.4f} | val_acc: {:.4f}"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        train_acc = 0.
        for X_batch, y_batch in tqdm(train_loader, leave=False, desc="Train"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            preds_labels = preds.argmax(dim=1)
            train_loss += loss.item()
            train_acc += accuracy_score(y_batch.cpu(), preds_labels.cpu())
            optimizer.zero_grad()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        lr_scheduler.step()

        model.eval()
        val_loss = 0.
        val_acc = 0.
        for X_batch, y_batch in tqdm(val_loader, leave=False, desc="Validation"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                preds_labels = preds.argmax(dim=1)
            val_loss += loss.item()
            val_acc += accuracy_score(y_batch.cpu(), preds_labels.cpu())
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        tqdm.write(log_template.format(epoch+1, train_loss, val_loss, train_acc, val_acc))
            



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset-path", type=str, required=True)
parser.add_argument("-b", "--batch-size", type=int, required=True)
parser.add_argument("-e", "--epochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("-o", "--out", type=str)

args = parser.parse_args()
args = vars(args)


files = [path for path in pathlib.Path(args.get("dataset_path")).rglob("*.jpg")]
train_files, val_files = train_test_split(files, test_size=0.2)
print(f"Total samples: {len(files)}\nTraining samples: {len(train_files)}\nValidation samples: {len(val_files)}")
label_encoder = LabelEncoder()
train_labels = [path.parent.name for path in train_files]
val_labels = [path.parent.name for path in val_files]
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
num_classes = len(label_encoder.classes_)
print(f"Training for {num_classes} classes")


train_dataset = CatsDogsDataset(train_files, train_labels)
val_dataset = CatsDogsDataset(val_files, val_labels)

model = CNN(num_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), args.get("lr"))
loss_fn = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.6)

train(model, optimizer, lr_scheduler, args.get("epochs"), args.get("batch_size"), loss_fn, DEVICE, train_dataset, val_dataset)

torch.save(model.state_dict, args.get("out", "./model.pt"))

