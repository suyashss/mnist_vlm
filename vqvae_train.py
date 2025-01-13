import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
from vqvae_model import VQVAE

model_save_file =  "vqvae_mnist_final_model.pth"

# Hyperparameters
input_dim = 28  # Input dimension (MNIST images are 28 x 28)
hidden_dim = 128  # Hidden layer size
latent_dim = 32  # Latent space dimensionality
quantization_embedding_dim = 4 # Dimension of vectors to be quantized 
num_embeddings = 26  # Number of embedding vectors in the codebook, set to 26 for A-Z
commitment_cost = 0.25  # Weight for the commitment loss
batch_size = 128  # Batch size for training
learning_rate = 2e-4  # Learning rate for optimizer
patience = 3 # For early stopping

# Set random seed 
pl.seed_everything(42, workers = True)

# Data loading and transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor
])

train_dataset_full = datasets.MNIST(root="./data", train=True, transform=transform, download=True)  # Load MNIST training dataset
train_dataset = Subset(train_dataset_full,range(50000))
val_dataset = Subset(train_dataset_full, range(50000,60000))

#Don't use MNIST validation dataset yet, we'll use that for LLM validation
#val_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)  # Load MNIST validation dataset

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # DataLoader for batching and shuffling
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # DataLoader for validation data

# Initialize Weights and Biases logger
wandb_logger = WandbLogger(project='vqvae-mnist')
wandb_logger.experiment.config['batch_size'] = batch_size
wandb_logger.experiment.config['patience'] = patience

# Model training using PyTorch Lightning
model = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings, 
              quantization_embedding_dim, commitment_cost, learning_rate)  # Initialize VQ-VAE model

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min')  

trainer = pl.Trainer(callbacks=[early_stopping], devices=1, 
                     accelerator='mps' if torch.backends.mps.is_available() else 'cpu', 
                     logger=wandb_logger, max_epochs=50)  # Initialize PyTorch Lightning trainer with W&B logger
trainer.fit(model, train_loader, val_loader)  # Train the model with training and validation data

# Save the final model to a file
torch.save(model.state_dict(),model_save_file)
