import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None

def grad_reverse(x, lambda_grl=1.0):
    return GradientReversalFunction.apply(x, lambda_grl)

class PartitionedEmbeddingModel(nn.Module):
    def __init__(self, input_dim, known_dim, hidden_dim, adv_output_dim=34):
        """
        Parameters:
          input_dim (int): Dimension of the input embeddings.
          known_dim (int): Dimension of the known latent space.
          hidden_dim (int): Dimension of the hidden layers.
          adv_output_dim (int): Number of classes for the adversarial task (default 20 for amino acid classification).
        """
        super(PartitionedEmbeddingModel, self).__init__()
        self.input_dim = input_dim
        self.known_dim = known_dim
        self.unknown_dim = input_dim - known_dim

        self.encoder_known = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.known_dim),
            nn.Tanh()
        )

        self.coeff_0_20 = nn.Parameter(torch.ones(1))
        self.coeff_20_23 = nn.Parameter(torch.ones(1)) 
        self.coeff_23_31 = nn.Parameter(torch.ones(1)) 
        self.coeff_31_32 = nn.Parameter(torch.ones(1)) 
        self.coeff_32_33 = nn.Parameter(torch.ones(1))

        self.encoder_unknown = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.unknown_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.known_dim + self.unknown_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(self.unknown_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, adv_output_dim)
        )

    def apply_specialized_layers(self, known_subspace):
        """
        Applies trainable coefficients to the known subspace at specified index ranges.
        """
        modified_known = known_subspace.clone() 

        modified_known[:, :20] *= self.coeff_0_20      # Range 0:20
        modified_known[:, 20:23] *= self.coeff_20_23     # Range 20:23
        modified_known[:, 23:31] *= self.coeff_23_31     # Range 23:31
        modified_known[:, 31:32] *= self.coeff_31_32     # Range 31:32
        modified_known[:, 32:33] *= self.coeff_32_33     # Range 32:33

        return modified_known

    def forward(self, x, lambda_grl=1.0):
        known_subspace = self.encoder_known(x)
        unknown_subspace = self.encoder_unknown(x)
        combined = torch.cat([known_subspace, unknown_subspace], dim=-1)
        known_subspace_with_coef = self.apply_specialized_layers(known_subspace)

        reconstructed = self.decoder(combined)

        reversed_unknown = grad_reverse(unknown_subspace, lambda_grl)
        adv_pred = self.discriminator(reversed_unknown)

        return known_subspace_with_coef, unknown_subspace, reconstructed, combined, adv_pred