import torch
import train
import argparse
import pandas as pd
import numpy as np
import seaborn as sn
import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help='Trained Model Path')
parser.add_argument("--test_data_path", type=str, help="Test Data Path")
parser.add_argument("--cmatrix_name", type = str, help = "CM map name")

class Evaluate():
    def __init__(self, datapath, checkpoint, imageName='output.png'):
        self.path = datapath
        self.checkpoint = checkpoint
        self.imageName = imageName
    def loadModel(self):
        model, loss_fn, optimizer, scheduler = train.get_model()
        model.load_state_dict(torch.load(self.checkpoint))
        print("Model Loaded Successfully!")
        return model
    def infer(self):
        model = self.loadModel()
        # -1 to use all images for testing
        testLoader = train.get_data(self.path, -1, 32, transforms.transforms_val)
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in tqdm(iter(testLoader)):
                output = model(inputs)
                output = output.data.cpu().numpy()
                output = [1.0 if p>=0.5 else 0.0 for p in output]
                y_pred.extend(output)

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)
        classes = ('NORMAL', 'PNEUMONIA')
        cf_matrix = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix", cf_matrix)
        print("Accuracy ", (cf_matrix[0,0]+cf_matrix[1,1])/(cf_matrix[0,0]+cf_matrix[0,1]+cf_matrix[1,0]+cf_matrix[1,1]))
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        plt.title("ResNet Pretrained Results with 0.1 LR Anneal")
        sn.heatmap(df_cm, annot=True)
        plt.savefig(self.imageName)
        print("Done")

args = parser.parse_args()
print("Arguments:",args)
eval = Evaluate(args.test_data_path, args.model_path, args.cmatrix_name)
eval.infer()


