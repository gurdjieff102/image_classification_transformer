import math
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from rotary_embedding_torch import RotaryEmbedding
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig, Trainer, TrainingArguments
from transformers.modeling_outputs import ImageClassifierOutput
from transformers import AutoConfig, AutoModel
from datasets import Dataset as HFDataset
import evaluate


class TransformerEncoderConfig(PretrainedConfig):
    model_type = "transformer_encoder"
    
    def __init__(
        self,
        dim=128,
        hidden=256,
        n_head=8,
        n_layers=10,
        n_class=10,
        dropout=0.1,
        patch_size=4,
        in_channels=1,
        image_size=28,
        **kwargs
    ):
        self.dim = dim
        self.hidden = hidden
        self.n_head = n_head
        self.n_layers = n_layers
        self.n_class = n_class
        self.dropout = dropout
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.image_size = image_size
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=1, out_channels=128):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)
        return x


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
        layers.append(RMSNorm(dim=dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

        self.seq = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.seq(x)
        return x


class TransformerEncoder(PreTrainedModel):
    config_class = TransformerEncoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [EncoderLayer(config.dim, config.hidden, config.n_head, config.dropout) 
             for i in range(config.n_layers)]
        )
        self.emb = LinearNet(config.image_size * config.image_size, config.dim)
        self.classifier = nn.Linear(config.dim, config.n_class)
        self.patch_emb = PatchEmbed(config.patch_size, config.in_channels, config.dim)
        self.conv_emb = ResNet18(config.in_channels, n_class=config.n_class)
        
        # Initialize weights
        self.post_init()

    def forward(self, pixel_values, labels=None):
        x = pixel_values
        
        # Linear embedding approach
        b, c, w, h = x.shape
        x = x.view(b, c, w * h)
        x = self.emb(x)
        x = x.mean(1)
        
        # Transformer approach (commented, but could be used instead)
        # x = self.patch_emb(x)
        # for layer in self.layers:
        #     x = layer(x)
        # x = x.mean(1)
        
        # Conv approach (commented, but could be used instead)  
        # x = self.conv_emb(x)
        
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )


# Register our custom model with HuggingFace
AutoConfig.register("transformer_encoder", TransformerEncoderConfig)
AutoModel.register(TransformerEncoderConfig, TransformerEncoder)


def create_hf_dataset(mnist_dataset):
    """Convert PyTorch MNIST dataset to HuggingFace dataset format"""
    def generator():
        for i in range(len(mnist_dataset)):
            image, label = mnist_dataset[i]
            # Convert tensor to numpy for HF dataset
            yield {
                'pixel_values': image.numpy(),
                'labels': label
            }
    
    return HFDataset.from_generator(generator)


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation"""
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def data_collator(features):
    """Custom data collator for our model"""
    batch = {}
    batch['pixel_values'] = torch.stack([torch.FloatTensor(f['pixel_values']) for f in features])
    batch['labels'] = torch.tensor([f['labels'] for f in features], dtype=torch.long)
    return batch


def main():
    # Model configuration
    config = TransformerEncoderConfig(
        dim=128,
        hidden=256,
        n_head=8,
        n_layers=10,
        n_class=10,
        dropout=0.1,
        patch_size=4,
        in_channels=1,
        image_size=28
    )

    # Initialize model
    model = TransformerEncoder(config)

    # Data preprocessing
    transform = transforms.Compose([transforms.ToTensor()])

    # Create PyTorch datasets first
    train_mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_mnist = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # Convert to HuggingFace datasets
    train_dataset = create_hf_dataset(train_mnist)
    val_dataset = create_hf_dataset(val_mnist)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard logging
        remove_unused_columns=False,  # Keep all columns
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Train the model
    print("Starting training with Hugging Face Trainer...")
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the model
    model.save_pretrained('./saved_model')
    config.save_pretrained('./saved_model')
    print("Model saved to ./saved_model")


if __name__ == "__main__":
    main()