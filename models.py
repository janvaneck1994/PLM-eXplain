import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None

def grad_reverse(x, lambda_grl=1.0):
    return GradientReversalLayer.apply(x, lambda_grl)

class SplitEmbeddingModel(nn.Module):
    def __init__(self, input_dim, known_dim, hidden_dim):
        super(SplitEmbeddingModel, self).__init__()
        self.input_dim = input_dim
        self.known_dim = known_dim
        self.unknown_dim = input_dim - known_dim
        
        # Encoder for known subspace
        self.encoder_known = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, known_dim),
            nn.Tanh()
        )
        
        # Encoder for unknown subspace
        self.encoder_unknown = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.unknown_dim),
            nn.Tanh()
        )
        
        # Decoder to reconstruct the input embedding
        self.decoder = nn.Sequential(
            nn.Linear(known_dim + self.unknown_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Discriminator to predict known subspace from unknown subspace
        self.discriminator = nn.Sequential(
            nn.Linear(self.unknown_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, known_dim)
        )
    
    def forward(self, x, lambda_grl=1.0):
        # Encode into known and unknown subspaces
        known_subspace = self.encoder_known(x)
        unknown_subspace = self.encoder_unknown(x)
        
        # Concatenate known and unknown subspaces
        combined = torch.cat([known_subspace, unknown_subspace], dim=-1)
        
        # Decode to reconstruct the input
        reconstructed = self.decoder(combined)
        
        # Apply GRL before discriminator
        reversed_unknown = grad_reverse(unknown_subspace, lambda_grl)
        discriminator_output = self.discriminator(reversed_unknown)
        
        return known_subspace, unknown_subspace, reconstructed, combined, discriminator_output
