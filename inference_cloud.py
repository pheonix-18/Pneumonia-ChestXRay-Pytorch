import cv2
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ChestXRay(nn.Module):
    classes = ['NORMAL', 'PNEUMONIA']
    def __init__(self, fpath = 'best_models/best_model.pth'):
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.transforms = T.Compose([T.ToPILImage(),
                             T.Resize((224,224)),
                              T.ToTensor(),
                              T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                              ])
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        self.model.load_state_dict(torch.load(fpath))
        print("\n----Loaded Model---")

    @torch.no_grad()
    def forward(self,x):
        with torch.no_grad():
            conf = self.model(x).item()
            pred = self.classes[round(conf)]
            return conf, pred

    def predict(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        x = self.transforms(img)
        x = x.unsqueeze(0)
        conf, pred = self.forward(x)
        return {'class' : pred, 'confidence' : f'{conf:.4f}'}





