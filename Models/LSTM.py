import torch.nn.functional as F

from Settings import *

class CharLSTM(nn.Module):
    def __init__(self):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(80, 256)
        self.lstm = nn.LSTM(256, 512, 2, batch_first=True)
        self.drop = nn.Dropout(p=0.05)
        self.out = nn.Linear(512, 80)
        self.to(device)

    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])






