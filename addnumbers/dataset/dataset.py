import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, n: int, length: int):
        self.input, self.target = self.generate_data(n ,length)


    def __getitem__(self, index: int) -> torch.Tensor:
        x = self.input[index]
        y = self.target[index]

        x = np.transpose(x, (1, 0))
        x = torch.from_numpy(np.array(x)).to(torch.float32)
        y = torch.from_numpy(np.array(y)).to(torch.float32)

        return x, y

    def __len__(self) -> int:
        return len(self.input)

    def generate_data(self, n: int, seq_length: int) -> np.array:

        x_num = np.random.uniform(0, 1, (n, 1, seq_length))
        x_mask = np.zeros([n, 1, seq_length])
        y = np.zeros([n, 1])
        for i in range(n):
            positions = np.random.choice(seq_length, size=2, replace=False)
            x_mask[i, 0, positions[0]] = 1
            x_mask[i, 0, positions[1]] = 1
            y[i, 0] = x_num[i, 0, positions[0]] + x_num[i, 0, positions[1]]
        x = np.concatenate((x_num, x_mask), axis=1)
        x = np.transpose(x, (0, 2, 1))

        return x, y
