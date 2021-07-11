import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--seq_len', type=int, default=450,
                    help='sequence length (default: )')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--hidden_layers', type=int, default=30,
                    help='number of hidden units per layer (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--saveCheckpoint', type=str, default='../checkpoint/450/',
                    help='Directory to store checkpoint')
parser.add_argument('--snapshots', type=int, default=20,
                    help='Snapshot Frequency of the checkpoint')

TRAIN_SAMPLES = 50000
VALIDATION_SAMPLES = 1000

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

train_set = CustomDataset(TRAIN_SAMPLES, seq_length)
validation_set = CustomDataset(VALIDATION_SAMPLES, seq_length)

train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=args.batch_size, shuffle=True)

model = TCN(input_channels, num_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_loss = []
validation_loss = []

def train(epoch):
    epoch_loss = 0

    model.train()
    for iteration, (input, label) in enumerate(train_loader):

        input = input.to(device)
        labels = label.to(device)

        t0 = time.time()
        prediction = model(input)
        loss = criterion(prediction, labels)
        t1 = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(train_loader), epoch_loss,
                                                                                 (t1 - t0)))
    train_loss.append(epoch_loss)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_set)))


def eval(epoch):
    with torch.no_grad():
        val_loss = 0
        for iteration, (input, label) in enumerate(validation_loader):
            input = input.to(device)
            labels = label.to(device)

            prediction = model(input)
            loss = criterion(prediction, labels)
            val_loss += loss.item()

        validation_loss.append(val_loss)
        print('Validation Loss of the model on the {} test samples: {} %'.format(len(validation_set), val_loss))

def checkpoint(epoch):
    if epoch % args.snapshots == 0:
        model_out_path = args.saveCheckpoint + "epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(0, args.epochs):
    train(epoch)
    eval(epoch)
    checkpoint(epoch)

plt.plot(range(args.epochs),train_loss)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.savefig('../results/train_loss_450.png')
plt.clf()

plt.plot(range(args.epochs),validation_loss)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.savefig('../results/val_loss_450.png')

