import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from transforms import *
from torchvision.models import resnet18
from dataset import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_model(pretrained=True, lr=0.01):
    print("Loading Pretrained Weights : ", pretrained)
    model = resnet18(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(in_features=128, out_features=1),
        nn.Sigmoid()
    )
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, threshold=0.001, verbose=True,
                                  min_lr=1e-5, threshold_mode='abs')
    return model.to(device), loss_fn, optimizer, scheduler

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_data(path, limit, batchSize, transforms):
    dataset = PneumoniaDataset(path=path, transforms=transforms, limit=limit)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=True, collate_fn=dataset.collate_fn)
    return loader


def train_epoch(model, loader, loss_fn, optimizer, scheduler):
    losses = []
    for batch in tqdm(iter(loader)):
        optimizer.zero_grad()
        x, y = batch
        y_ = model(x)
        loss = loss_fn(y_.permute(1,0).squeeze(0), y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    return sum(losses) / len(losses)


def val_epoch(model, loader, loss_fn, optimizer, scheduler):
    losses = []
    with torch.no_grad():
        for batch in tqdm(iter(loader)):
            optimizer.zero_grad()
            x, y = batch
            y_ = model(x)
            loss = loss_fn(y_.permute(1,0).squeeze(0), y)
            losses.append(loss.item())
            scheduler.step(loss.item())
    return sum(losses) / len(losses)


def validate(model, loader):
    acc = []
    with torch.no_grad():
        for batch in tqdm(iter(loader)):
            x, y = batch
            pred = model(x)
            acc.append(accuracy(pred, y))
    return sum(acc) / len(acc)


def accuracy(y_, y):
    correct = 0
    for i in range(len(y_)):
        if y_[i]>=0.5:
            pred = 1
        else:
            pred = 0
        if pred == y[i]:
            correct += 1

    return correct / len(y_)


def train(model, trn_dl, val_dl, loss_fn, optimizer, scheduler, startEpoch, endEpoch, writer, best_acc, args):
    for epoch in tqdm(range(startEpoch, endEpoch)):
        train_loss = train_epoch(model, trn_dl, loss_fn, optimizer, scheduler)
        train_acc = validate(model, trn_dl)
        val_acc = validate(model, val_dl)
        val_loss = val_epoch(model, val_dl, loss_fn, optimizer, scheduler)
        lr = get_lr(optimizer)

        PATH = ["./saved_models/"+str(args.experimentName)+"/model", str(epoch), str(args.batchSize),  str(val_acc), str(val_loss), str(lr),".pth"]
        PATH = '_'.join(PATH)
        print("Saving Model to Path", PATH)
        torch.save(model.cpu().state_dict(), PATH)

        print("Epoch {} | Training Loss {} | Training Accuracy {}| Validation Loss {}| Validation Accuracy {} | LR {}".format(
            epoch, train_loss, train_acc, val_loss, val_acc, lr))
        writer.add_scalars('Losses', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
