import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Encoder Module
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # Fully connected layers to reduce dimensionality
        self.fc1 = nn.Linear(input_dim * input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, latent_dim)  # Second fully connected layer
        self.input_dim = input_dim
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        return x 

# Decoder Module
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # Fully connected layers to reconstruct the input
        self.fc1 = nn.Linear(latent_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim * output_dim)  # Second fully connected layer
        self.output_dim = output_dim
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input to correct shape
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid activation to output layer to keep values between 0 and 1
        return x.view(x.size(0), 1, self.output_dim, self.output_dim)  # Reshape to match image dimensions

# Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, quantization_embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.quantization_embedding_dim = quantization_embedding_dim
        self.commitment_cost = commitment_cost

        # Embedding layer that contains the codebook vectors
        self.embedding = nn.Embedding(self.num_embeddings, self.quantization_embedding_dim)
        self.initialized = False

    def initialize_codebook(self, x):
        flat_x = x.view(-1, self.quantization_embedding_dim).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_embeddings, random_state=0).fit(flat_x)
        self.embedding.weight.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(x.device)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize_codebook(x)  # Initialize codebook with k-means on the first batch
        
        # Flatten input from (batch_size, latent_dim) to (batch_size * (latent_dim/embedding_dim), embedding_dim)
        flat_x = x.view(-1, self.quantization_embedding_dim)
        
        # Calculate distances between input vectors and embedding vectors
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        
        # Get the indices of the closest embedding vector for each input vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Get the quantized vectors using the indices
        quantized = self.embedding(encoding_indices).view(x.shape)
        
        # Calculate loss to ensure embeddings commit to the quantized values
        e_latent_loss = F.mse_loss(quantized.detach(), x)  # Loss for the encoder to match the quantized vector
        q_latent_loss = F.mse_loss(quantized, x.detach())  # Loss for the quantized vector to match the encoder output
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Pass gradients to encoder using the straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, loss, encoding_indices

# VQ-VAE Model
class VQVAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings, quantization_embedding_dim, commitment_cost, learning_rate):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)  # Encoder to produce latent representation
        self.quantizer = VectorQuantizer(num_embeddings, quantization_embedding_dim, commitment_cost)  # Vector quantizer to discretize the latent representation
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)  # Decoder to reconstruct the input from the quantized representation
        self.learning_rate = learning_rate
        self.latents = []
        self.labels = []

        self.save_hyperparameters()

    def forward(self, x):
        z = self.encoder(x)  # Encode input to latent representation
        quantized, vq_loss, encoding_indices = self.quantizer(z)  # Quantize the latent representation
        x_recon = self.decoder(quantized)  # Decode the quantized representation to reconstruct the input
        return x_recon, vq_loss, z, encoding_indices

    def training_step(self, batch, batch_idx):
        data, _ = batch
        recon_data, vq_loss, _, encoding_indices = self(data)  # Forward pass
        recon_loss = F.mse_loss(recon_data, data)  # Reconstruction loss (MSE between input and reconstruction)
        loss = recon_loss + vq_loss  # Total loss is the sum of reconstruction loss and VQ loss
        self.log_dict({'train_loss': loss, 'recon_loss': recon_loss, 'quant_loss': vq_loss})  # Log training loss

        # For the 0th batch, return the counts of each index from the codebook that the batch examples are mapped to
        if batch_idx == 0:
            unique_indices, counts = torch.unique(encoding_indices, return_counts=True)
            print(counts)
            self.log_dict({f'codebook_index_{i}': count for i, count in zip(unique_indices.tolist(), counts.tolist())})

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        recon_data, vq_loss, z, _ = self(data)  # Forward pass
        recon_loss = F.mse_loss(recon_data, data)  # Reconstruction loss
        loss = recon_loss + vq_loss  # Total loss
        self.log('val_loss', loss)  # Log validation loss
        
        # Collect latent representations for t-SNE visualization
        if batch_idx % 5 == 0:  # Collect data from 20% of validation batches for visualization
            self.latents.append(z.view(data.size(0), -1).cpu().numpy())
            self.labels.append(target.cpu().numpy())

    def on_validation_epoch_end(self):
        # Run t-SNE visualization at the end of each epoch using 1/5th of the validation set
        latents = np.concatenate(self.latents, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        
        tsne = TSNE(n_components=2, random_state=42)  # Apply t-SNE to the latent representations
        latents_2d = tsne.fit_transform(latents)
        
        # Plot the t-SNE results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Digit Label')
        plt.title("t-SNE Visualization of Latent Representations")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        img_path = "tsne_visualization_epoch_{}.png".format(self.current_epoch)
        plt.savefig(img_path)  # Save plot instead of showing 
        plt.close()
        
        # Log plot to wandb
        wandb.log({"clustering_examples": wandb.Image(img_path)})

        # Clear the latents and labels after each epoch
        self.latents.clear()
        self.labels.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
