"""Models for evidential deep learning using PyTorch."""

from __future__ import annotations

import torch
from torch import nn

from probly.layers.evidential.torch import (
    EncoderMnist,
    EvidentialHead,
    MLPEncoder,
    RadialFlowDensity,
    SimpleHead,
)


class NatPN(nn.Module):
    """Natural Posterior Network for evidential deep learning with normalizing flows.

    Combines encoder, normalizing flow density, and head for uncertainty quantification.
    Users can provide custom encoders for different data modalities.

    Args:
        encoder: Encoder module mapping raw inputs to latent space.
        head: Head module (DirichletHead for classification, GaussianHead for regression).
        latent_dim: Dimension of the latent space.
        flow_length: Number of radial flow layers. Defaults to 4.
        certainty_budget: Budget for certainty calibration. If None, defaults to latent_dim.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        latent_dim: int,
        flow_length: int = 4,
        certainty_budget: float | None = None,
    ) -> None:
        """Initialize the NatPN model."""
        super().__init__()

        self.encoder = encoder
        self.head = head

        self.flow = RadialFlowDensity(
            dim=latent_dim,
            flow_length=flow_length,
        )

        if certainty_budget is None:
            certainty_budget = float(latent_dim)
        self.certainty_budget = certainty_budget

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (for transfer learning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights (for fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through encoder, flow, and head.

        Args:
            x: Input tensor compatible with the encoder.

        Returns:
            Dictionary with predictions from the head (including alpha for classification,
            or mean/var for regression) along with latent space information.
        """
        z = self.encoder(x)  # [B, latent_dim]
        log_pz = self.flow.log_prob(z)  # [B]

        return self.head(
            z=z,
            log_pz=log_pz,
            certainty_budget=self.certainty_budget,
        )


class DirichletHead(nn.Module):
    """Dirichlet posterior head for evidential classification.

    Takes latent representations and outputs Dirichlet parameters for uncertainty
    quantification in classification. This head should be used with an encoder to
    create a complete classification model.

    Args:
        latent_dim: Dimension of input latent vectors from an encoder.
        num_classes: Number of classification classes.
        hidden_dim: Dimension of hidden layer. Defaults to 64.
        n_prior: Prior for evidence scaling. If None, defaults to num_classes.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        n_prior: float | None = None,
    ) -> None:
        """Initialize the DirichletHead."""
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        if n_prior is None:
            n_prior = float(num_classes)

        chi_prior = torch.full((num_classes,), 1.0 / num_classes)
        alpha_prior = n_prior * chi_prior

        self.register_buffer("alpha_prior", alpha_prior)

    def forward(
        self,
        z: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Compute Dirichlet parameters for evidential classification.

        Args:
            z: Latent representations of shape [B, latent_dim].
            log_pz: Log probability from density estimator of shape [B].
            certainty_budget: Budget parameter for evidence scaling.

        Returns:
            Dictionary containing:
                - alpha: Dirichlet parameters [B, num_classes]
                - z: Input latent representations
                - log_pz: Log density values
                - evidence: Scaled evidence [B, num_classes]
        """
        logits = self.classifier(z)  # [B, C]
        chi = torch.softmax(logits, dim=-1)  # [B, C]

        # Total evidence n(x)
        n = certainty_budget * log_pz.exp()  # [B]
        n = torch.clamp(n, min=1e-8)

        evidence = n.unsqueeze(-1) * chi  # [B, C]
        alpha = self.alpha_prior.unsqueeze(0) + evidence

        return {
            "alpha": alpha,  # Dirichlet parameters
            "z": z,
            "log_pz": log_pz,
            "evidence": evidence,
        }


class GaussianHead(nn.Module):
    """Gaussian posterior head for evidential regression.

    Takes latent representations and outputs mean and variance for Gaussian
    uncertainty quantification in regression. This head should be used with an encoder
    to create a complete regression model.

    Args:
        latent_dim: Dimension of input latent vectors from an encoder.
        out_dim: Dimension of regression output. Defaults to 1 (univariate regression).
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 1,
    ) -> None:
        """Initialize the GaussianHead."""
        super().__init__()

        self.mean_net = nn.Linear(latent_dim, out_dim)
        self.log_var_net = nn.Linear(latent_dim, out_dim)

    def forward(
        self,
        z: torch.Tensor,
        log_pz: torch.Tensor,
        certainty_budget: float,
    ) -> dict[str, torch.Tensor]:
        """Compute Gaussian parameters for evidential regression.

        Args:
            z: Latent representations of shape [B, latent_dim].
            log_pz: Log probability from density estimator of shape [B].
            certainty_budget: Budget parameter for precision scaling.

        Returns:
            Dictionary containing:
                - mean: Predicted mean [B, out_dim]
                - var: Predicted variance [B, out_dim]
                - z: Input latent representations
                - log_pz: Log density values
                - precision: Scaled precision [B, out_dim]
        """
        mean = self.mean_net(z)  # [B, D]
        log_var = self.log_var_net(z)  # [B, D]

        # Epistemic uncertainty via density scaling
        precision = certainty_budget * log_pz.exp().unsqueeze(-1)
        precision = torch.clamp(precision, min=1e-8)

        var = torch.exp(log_var) / precision

        return {
            "mean": mean,
            "var": var,
            "z": z,
            "log_pz": log_pz,
            "precision": precision,
        }


class SimpleCNN(nn.Module):
    """Simple CNN model for evidential classification."""

    def __init__(  # noqa: D107
        self,
        encoder: nn.Module | None = None,
        head: nn.Module | None = None,
        latent_dim: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if encoder is None:
            encoder = EncoderMnist(latent_dim=latent_dim)

        if head is None:
            head = SimpleHead(latent_dim=latent_dim, num_classes=num_classes)

        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        z = self.encoder(x)
        return self.head(z)


class EvidentialRegressionModel(nn.Module):
    """Full evidential regression model combining encoder and evidential head."""

    def __init__(self, encoder: nn.Module | None = None) -> None:
        """Initialize the full model."""
        super().__init__()
        if encoder is None:
            encoder = MLPEncoder(feature_dim=32)

        self.encoder = encoder
        self.head = EvidentialHead(feature_dim=encoder.feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and head."""
        features = self.encoder(x)
        return self.head(features)
