import math
import torch
from torch import nn
import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from rotary_embedding_torch import RotaryEmbedding
import torch.optim as optim


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.gamma

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=1, out_channels=128):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)
        b, c, h ,w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)
        return x 
    
# x = torch.randn(32, 1, 28, 28).to(device)
# patch_emb = PatchEmbed().to(device)
# token = patch_emb(x)
# print(token.shape)

class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert dim % n_head == 0, 'dim % n_head == 0'
        self.n_d = dim // n_head
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.rot_emb = RotaryEmbedding(self.n_d)


    def forward(self, x, mask=None):
        b, c, t = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(b, c, self.n_head, self.n_d).permute(0, 2, 1, 3)
        k = k.view(b, c, self.n_head, self.n_d).permute(0, 2, 1, 3)
        v = v.view(b, c, self.n_head, self.n_d).permute(0, 2, 1, 3)

        q = self.rot_emb.rotate_queries_or_keys(q)
        k = self.rot_emb.rotate_queries_or_keys(k)
        atten = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_d)
        if mask is not None:
            mask = mask.to(atten.device)
            atten = atten.masked_fill(mask==0, float('-inf'))

        atten = self.softmax(atten)
        atten = self.dropout(atten)
        atten = torch.matmul(atten, v)
        out = atten.permute(0, 2, 1, 3).contiguous().view(b, c, t)
        out = self.out_proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )
       
    def forward(self, x):
        x = self.ff(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, dim, hidden, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.atten = MultiheadAttention(dim, n_head)
        self.ff = FeedForward(dim, hidden)
        self.norm1 = RMSNorm(dim=dim)
        self.norm2 = RMSNorm(dim=dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1]))
        atten = self.atten(x, mask=mask)
        x = self.norm1(x + self.dropout(atten))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, downsample=None):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        out = self.relu(x)
        # out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out = out + x 
        out = self.relu(x + out)
        return out 
    
class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels=64, stride=2, padding=3, n_class=10):
        super(ResNet18, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__make_layer(64, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, n_class)

    def __make_layer(self, out_channels, blocks, stride=1):
        layers = []
        for _ in range(blocks):
           layers.append(ConvBlock(self.out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class LinearNet(nn.Module):
    def __init__(self, in_dim, dim, dropout=0.1):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(in_dim, dim)
        layers = []
        layers.append(nn.Linear(dim, dim))
        # layers.append(nn.LayerNorm(dim))
        layers.append(RMSNorm(dim=dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        self.seq = nn.Sequential(*layers)
    def forward(self, x):
        x = self.linear(x)
        x = self.seq(x)
        return x
    
class TransformerEncoder(L.LightningModule):
    def __init__(self, dim, hidden, n_head, n_layers, n_class, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(dim, hidden, n_head, dropout) for i in range(n_layers)]
        )
        self.emb = LinearNet(28*28, dim)
        self.fc = nn.Linear(dim, n_class)
        self.loss_fn = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch_size = 256

        self.patch_emb = PatchEmbed(4, 1, 128)
        self.conv_emb = ResNet18(1)
    def forward(self, x):
        # linear 
        b, c, w, h = x.shape
        x = x.view(b, ｃ, w *ｈ)
        x = self.emb(x)
        x = x.mean(1)

        # transformer 
        # x = self.patch_emb(x)
        # for layer in self.layers:
        #     x = layer(x)
        # x = x.mean(1)
        # x = self.fc(x)

        # conv
        # x = self.conv_emb(x)

        return x

    def training_step(self, batch, idx):
        x, y = batch
        x = self(x)
        loss = self.loss_fn(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        pass
    
    def validation_step(self, batch, idx):
        x, y = batch
        x = self(x)
        loss = self.loss_fn(x, y)
        pred = torch.argmax(x, dim=-1)
        acc = (pred == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('acc', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def train_dataloader(self):
        train_ds = datasets.MNIST(root='./data',
                                train=True,
                                download=True,
                                transform=self.transform)
        
        dl = DataLoader(dataset=train_ds,
                        shuffle=True,
                        batch_size=self.batch_size)
        return dl
    
    def val_dataloader(self):
        val_ds = datasets.MNIST(root='./data',
                                train=False,
                                download=True,
                                transform=self.transform)
        
        vl = DataLoader(dataset=val_ds,
                        shuffle=False,
                        batch_size=self.batch_size)
        return vl

dim = 128
n_layers = 10
dropout = 0.1
n_class = 10
n_head = 8

model = TransformerEncoder(dim, dim * 2, n_head, n_layers, n_class, dropout)

trainer = L.Trainer(
    max_epochs=4,
    accumulate_grad_batches=4,
    accelerator='gpu' if torch.cuda.is_available else 'cpu',
    devices='auto'
)

trainer.fit(model)


