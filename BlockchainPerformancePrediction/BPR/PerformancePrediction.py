import pickle
from argparse import ArgumentParser
from tqdm import *
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn import preprocessing

torch.set_float32_matmul_precision('high')
Dataset = ""

class Backbone(torch.nn.Module):
    global Dataset
    def __init__(self):
        super().__init__()
        if Dataset == "BPD-1" or "MMBPD":
            self.fc1 = nn.Linear(2, 128)
        else:
            self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MMBPD", help='Indicates which data set to use')
    parser.add_argument('--task', type=str, default="Throughput", help='Indicates which task to perform')
    args = parser.parse_args()
    global Dataset
    Dataset = args.dataset

    raw_data = pd.read_csv("../data/" + str(args.dataset) + '.csv').values
    if args.dataset == "BPD-1":
        X = raw_data[:, :2]
        latency = [i[2] for i in raw_data]
        throughput = [i[3] for i in raw_data]
    elif args.dataset == "HFBTP":
        X = raw_data[:, :3]
        latency = [i[4] for i in raw_data]
        throughput = [i[3] for i in raw_data]
    elif args.dataset == "MMBPD":
        X = raw_data[:, 1:3]
        latency = [i[4] for i in raw_data]
        throughput = [i[-2] for i in raw_data]
        X = raw_data[:,3:5]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(X)

    MLP = Backbone()
    backbone = torch.load("model/"+str(args.dataset)+"/Throughput/MLP_Throughput1.ckpt", weights_only=True)
    MLP.load_state_dict(backbone)
    Throughput = []

    MLP.eval()
    with torch.no_grad():
        for content in tqdm(X):
            transactiontraffic = content[0]
            blocksize = content[1]
            if args.dataset == "HFBTP":
                orderer = content[2]
                input = np.array([transactiontraffic, blocksize, orderer]).reshape(-1, 3)
            else:
                input = np.array([transactiontraffic, blocksize]).reshape(-1, 2)
            input = torch.from_numpy(min_max_scaler.transform(input)).type(torch.float32)
            prediction = MLP(input)
            Throughput.append(prediction.item())

    args.task = "Latency"
    MLP = Backbone()
    backbone = torch.load("model/" + str(args.dataset) + "/Throughput/MLP_Latency1.ckpt", weights_only=True)
    MLP.load_state_dict(backbone)
    Latency = []

    MLP.eval()
    with torch.no_grad():
        for content in tqdm(X):
            transactiontraffic = content[0]
            blocksize = content[1]
            if args.dataset == "HFBTP":
                orderer = content[2]
                input = np.array([transactiontraffic, blocksize, orderer]).reshape(-1, 3)
            else:
                input = np.array([transactiontraffic, blocksize]).reshape(-1, 2)
            input = torch.from_numpy(min_max_scaler.transform(input)).type(torch.float32)
            prediction = MLP(input)
            Latency.append(prediction.item())

    if args.dataset == "HFBTP":
        output = pd.concat([pd.Series(X[:, 0], name='Transaction Arrival Rate'),
                            pd.Series(X[:, 1], name='Block Size'), pd.Series(X[:, 2], name='Orderers'), pd.Series(latency, name='Latency'),
                            pd.Series(throughput, name='Throughput'),
                            pd.Series(Latency, name='Latency_Pred'),
                            pd.Series(Throughput, name='Throughput_Pred')], axis=1)
        output.to_csv('HFBTP_LearningChain.csv')
    else:
        output = pd.concat([pd.Series(X[:, 0], name='Transaction Arrival Rate'),
                            pd.Series(X[:, 1], name='Block Size'), pd.Series(latency, name='Latency'),
                            pd.Series(throughput, name='Throughput'),
                            pd.Series(Latency, name='Latency_Pred'),
                            pd.Series(Throughput, name='Throughput_Pred')], axis=1)
        output.to_csv(str(args.dataset)+'_LearningChain.csv')



if __name__ == '__main__':
    cli_main()
