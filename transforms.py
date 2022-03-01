from torchvision.transforms.transforms import RandomHorizontalFlip
from torchvision.transforms.functional import hflip
import torch
from torchvision import transforms as T


transforms_train = T.Compose([T.ToPILImage(),
                              T.Resize((224,224)),
                              T.RandomHorizontalFlip(p=0.5),
                              T.ColorJitter(brightness=(0.95,1.05)),
                              T.RandomRotation(5),
                              T.ToTensor(),
                              T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                              ])
transforms_val = T.Compose([T.ToPILImage(),
                             T.Resize((224,224)),
                              T.ToTensor(),
                              T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                              ])
