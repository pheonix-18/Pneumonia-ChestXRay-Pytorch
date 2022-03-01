import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
import dataset
import train
import os
import transforms
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--trainPath",type=str, default='./Data/train')
parser.add_argument("--valPath",type=str, default='./Data/val')
parser.add_argument("--batchSize", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--endEpoch", type=int, default= 10)
parser.add_argument("--startEpoch", type=int, default = 0)
parser.add_argument("--optimizer", type=str, default='adam', choices=['adam','sgd'])
parser.add_argument("--loss_fn", type=str, default='bce',choices=['bce'])
parser.add_argument("--pretrained", dest='pretrained', default=False, action="store_true")
parser.add_argument("--experimentName", default='ResNet', type=str)
# runs/experimentName
# saved_models/experimentName/
parser.add_argument("--lrRateAnnealing", default=0.1, type=float)
parser.add_argument("--resume", dest='resume', help="Resume Training", default=False, action='store_true')
parser.add_argument("--checkpoint", type=str, help='Model checkpoint')
parser.add_argument("--trainLimit", type=int, default = 5216, help="Dataset size")
parser.add_argument("--valLimit", type=int, default = 16)

parser = parser.parse_args()
print(parser)

print("====Loading Dataset ====")
train_dl = train.get_data(parser.trainPath, parser.trainLimit, parser.batchSize, transforms.transforms_train)
val_dl = train.get_data(parser.valPath, parser.valLimit, parser.batchSize, transforms.transforms_val)
print("Data Loader Ready")

# Show a sample Image
# img, label = next(iter(train_dl))
# print(img.shape)
# im = img[0].permute(1,2,0)
# plt.imshow(im.numpy())
# plt.show()

print("====Loading Model ====")
model, loss_fn, optimizer, scheduler = train.get_model(parser.pretrained, parser.lr)
best_acc = 0

if parser.resume:
    print("===Loading Saved Model===")
    model.load_state_dict(torch.load(parser.checkpoint))
    model.to(device)
    best_acc = parser.checkpoint.split("/")[-1].split('_')[3]
    parser.lr = float(parser.checkpoint.split("/")[-1].split('_')[5])
    optimizer = torch.optim.Adam(model.parameters(), lr=parser.lr)
print("Model Loaded")
print("==== Starting Training ====")
SWriterPATH = './runs/'+str(parser.experimentName)+"_"+str(parser.batchSize)+"_"+str(parser.lr)+"/"
writer = SummaryWriter(SWriterPATH)
saved_models_directory = "./saved_models/"+str(parser.experimentName)+"/"
os.makedirs(saved_models_directory,exist_ok=True)
train.train(model, train_dl, val_dl, loss_fn, optimizer, scheduler, parser.startEpoch, parser.endEpoch, writer, best_acc, parser)
print("Training Done!")
#cv2.imshow("Sample", )





