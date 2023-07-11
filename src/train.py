import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from dataset import PokemonDataset
from torchvision.transforms import ToTensor, Compose, Resize, ColorJitter, RandomAffine, CenterCrop, Normalize
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import os
import shutil
from tqdm.autonotebook import tqdm
import torchvision.models as models
from categories import categories
from utils import get_args
torch.cuda.empty_cache()

def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device")

    num_epochs = args.epochs
    batch_size = args.batch_size
    # train_set = CIFAR10(root="data", train=True, download=True, transform=ToTen5sor())
    train_transform = Compose([
        Resize((224, 224)),
        CenterCrop(size=200),
        RandomAffine(degrees=(-1, 1), scale=(0.9, 1.1), center=(0.5, 0.5)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.5),
        ToTensor(),
        Normalize(mean=[0.59565047, 0.59168748, 0.56110077], std=[0.23401408, 0.22395668, 0.23227972])
    ])

    train_set = PokemonDataset(root="../data", train=True, transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=13,
        drop_last=False,
    )
    # val_set = CIFAR10(root="data", train=False, download=True, transform=ToTensor())
    val_transform = Compose([
        Resize(size=(224, 224)),
        ToTensor(),
        Normalize(mean=[0.59565047, 0.59168748, 0.56110077], std=[0.23401408, 0.22395668, 0.23227972])
    ])
    val_set = PokemonDataset(root="../data", train=False, transform=val_transform)
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=13,
        drop_last=False,
    )

    model = models.densenet121()
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(categories))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.001)
    num_iters = len(train_dataloader)

    model.to(device)

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    best_acc = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            writer.add_scalar("Train/Loss", np.mean(train_loss), num_iters*epoch+iter)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, num_epochs, np.mean(train_loss)))

        model.eval()
        all_labels = []
        all_predictions = []

        for iter, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                # with torch.inference_model():
                output = model(images)
                _, predictions = torch.max(output, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Val/Accuracy", acc, epoch)
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pt"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pt"))
            best_acc = acc


if __name__ == '__main__':
    args = get_args()
    train(args)
