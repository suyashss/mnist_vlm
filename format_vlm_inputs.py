import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
from vqvae_model import VQVAE
import os

outdir = "nanoGPT/data/mnist_vlm"
os.mkdir(outdir)

# Step 1: Load the model
# Hyperparameters
input_dim = 28  # Input dimension (MNIST images are 28x28)
hidden_dim = 128  # Hidden layer size
latent_dim = 32  # Latent space dimensionality
num_embeddings = 26  # Number of embedding vectors in the codebook
quantization_embedding_dim = 4 # Dimension of vectors to be quantized 
commitment_cost = 0.25  # Weight for the commitment loss
learning_rate = 2e-4  # Learning rate for optimizer

model = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings, 
                quantization_embedding_dim,commitment_cost, learning_rate)  # Initialize VQ-VAE model
model.load_state_dict(torch.load("vqvae_mnist_final_model.pth",
                                 weights_only=True))
model.eval()
model.quantizer.initialized = True

# Define the transform for MNIST images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the MNIST dataset
train_dataset_full = datasets.MNIST(root="./data", train=True, transform=transform, download=True)  # Load MNIST training dataset
train_dataset = Subset(train_dataset_full,range(50000))
val_dataset = Subset(train_dataset_full, range(50000,60000))

#Use MNIST validation dataset for VLM testing
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)  # Load MNIST validation dataset

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # DataLoader for batching and shuffling
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # DataLoader for validation data
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # DataLoader for test data

# img2text function that creates a text description from image token indices
def img2text(representation):
    letters = []
    for index in representation:
        if 0 <= index <= 25:
            letters.append(chr(ord('A') + index))
        else:
            raise ValueError("Index out of range. Indices should be between 0 and 25.")
    return "".join(letters)

# Iterate over the data and generate text sentences
def save_dataset_txt(data_loader,outfile,maxlen=64):
    with open(outfile,"w") as ofile:
        for idx, (image, label) in enumerate(data_loader):
            # Get the output from the model (encode/decode or latent representation)
            with torch.no_grad():
                # Get image representation as codebook indices
                _,_,_,representation = model(image)  
                img_tokens = img2text(representation)
            # Generate a sentence with 50% probability for each type
            if random.random() < 0.5:
                sentence = f"number:{label.item()}, image:{img_tokens}."
            else:
                sentence = f"image:{img_tokens}, number:{label.item()}."

            if len(sentence) < maxlen:
                sentence += ("#"*(maxlen - len(sentence)))
            # Print or store the generated sentence
            ofile.write(sentence+"\n")

save_dataset_txt(train_loader,os.path.join(outdir,"train.txt"),32)
save_dataset_txt(val_loader,os.path.join(outdir,"valid.txt"),32)
save_dataset_txt(test_loader,os.path.join(outdir,"test.txt"),32)
