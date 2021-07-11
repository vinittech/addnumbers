import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import time
import matplotlib.pyplot as plt

from addnumbers.dataset import CustomDataset
from addnumbers.model.rnn import TCN

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=150,
                    help='sequence length (default: 150)')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--hidden_layers', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--checkpoint', type=str, default='../checkpoint/epoch_160.pth',
                    help='Directory to store checkpoint')
parser.add_argument('--snapshots', type=int, default=100,
                    help='Snapshot Frequency of the checkpoint')


args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_channels = 2
num_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len

channel_sizes = [args.hidden_layers]*args.levels
kernel_size = args.ksize
dropout = args.dropout

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TCN(input_channels, num_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
criterion = nn.MSELoss()

TEST_SAMPLES = 40
input, target = CustomDataset.generate_data(TEST_SAMPLES, args.seq_len)

error = 0
for index in range(TEST_SAMPLES):

    tensor = torch.from_numpy(input[index]).to(torch.float32).to(device).unsqueeze(0)
    prediction = model(tensor).item()
    error += np.square(np.subtract(prediction, target[index]))
    
print('The Average loss over {} Samples is {}'.format(TEST_SAMPLES, error/TEST_SAMPLES))