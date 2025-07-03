import math
from einops import repeat, rearrange
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from tqdm import tqdm
from rotary_embedding_torch import RotaryEmbedding
import lightning as L
import torch.optim as optim
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, dim, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert dim % n_head == 0, 'dim % n_head != 0'
        self.n_head = n_head
        self.dim = dim
        self.n_d = dim // n_head
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.rot_emb = RotaryEmbedding(self.n_d)

    def forward(self, x, mask=None):
        b, t, d = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_head, self.n_d).permute(0, 2, 1, 3)
        k = k.view(b, t, self.n_head, self.n_d).permute(0, 2, 1, 3)
        v = v.view(b, t, self.n_head, self.n_d).permute(0, 2, 1, 3)

        q = self.rot_emb.rotate_queries_or_keys(q)
        k = self.rot_emb.rotate_queries_or_keys(k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_d)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, d)
        out = self.out_proj(out)
        return out
  
class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout=0.1):
    super(FeedForward, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, dim, n_head, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(n_head=n_head, dim=dim, dropout=dropout)
        self.ff = FeedForward(dim, hidden_dim=hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(x + self.dropout(attention))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class Encoder(L.LightningModule):
    def __init__(self, n_layers, dim, n_head, hidden_dim, n_class, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(dim=dim, n_head=n_head, hidden_dim=hidden_dim, dropout=dropout) for _ in range(n_layers)]
        )
        #   self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.embed = nn.Linear(1, dim)
        self.fc = nn.Linear(dim, n_class)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(1)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, idx):
        x, y = batch
        b, c, h, w = x.shape
        x = x.view(b, h * w, 1)
        x = self.embed(x)
        x = self(x)
        
        loss = self.loss_fn(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss 

    def validation_step(self, batch, idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        b, c, h, w = x.shape
        x = x.view(b, h * w, 1)
        x = self.embed(x)
        pre = self(x)
        loss = self.loss_fn(pre, y)
        pres = torch.argmax(pre, dim=-1)
        acc = (pres == y).float().mean()
        

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}


    def configure_optimizers(self):
       return optim.Adam(self.parameters(), lr=0.001)

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transform)

batch_size = 32
dl = DataLoader(dataset=train_ds,
                shuffle=True,
                batch_size=batch_size)

test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
tl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = Encoder(2, 128, 8, 8, 10)
trainer = L.Trainer(
   max_epochs=5,
   accelerator='gpu' if torch.cuda.is_available() else 'cpu',
   devices='auto'
)
trainer.fit(model, train_dataloaders=dl, val_dataloaders=tl)













