"""Collection of torch evidential training functions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EvidentialLogLoss(nn.Module):
    """Evidential Log Loss based on :cite:`sensoyEvidentialDeep2018`."""

    def __init__(self) -> None:
        """Intialize an instance of the EvidentialLogLoss class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential log loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        loss = torch.mean(torch.log(strengths) - torch.log(alphas[torch.arange(targets.shape[0]), targets]))
        return loss


class EvidentialCELoss(nn.Module):
    """Evidential Cross Entropy Loss based on :cite:`sensoyEvidentialDeep2018`."""

    def __init__(self) -> None:
        """Intialize an instance of the EvidentialCELoss class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential cross entropy loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        loss = torch.mean(torch.digamma(strengths) - torch.digamma(alphas[torch.arange(targets.shape[0]), targets]))
        return loss


class EvidentialMSELoss(nn.Module):
    """Evidential Mean Square Error Loss based on :cite:`sensoyEvidentialDeep2018`."""

    def __init__(self) -> None:
        """Intialize an instance of the EvidentialMSELoss class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential mean squared error loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        strengths = torch.sum(alphas, dim=1)
        y = F.one_hot(targets, inputs.shape[1])
        p = alphas / strengths[:, None]
        err = (y - p) ** 2
        var = p * (1 - p) / (strengths[:, None] + 1)
        loss = torch.mean(torch.sum(err + var, dim=1))
        return loss


class EvidentialKLDivergence(nn.Module):
    """Evidential KL Divergence Loss based on :cite:`sensoyEvidentialDeep2018`."""

    def __init__(self) -> None:
        """Initialize an instance of the EvidentialKLDivergence class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential KL divergence loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        alphas = inputs + 1.0
        y = F.one_hot(targets, inputs.shape[1])
        alphas_tilde = y + (1 - y) * alphas
        strengths_tilde = torch.sum(alphas_tilde, dim=1)
        k = torch.full((inputs.shape[0],), inputs.shape[1], device=inputs.device)
        first = torch.lgamma(strengths_tilde) - torch.lgamma(k) - torch.sum(torch.lgamma(alphas_tilde), dim=1)
        second = torch.sum(
            (alphas_tilde - 1) * (torch.digamma(alphas_tilde) - torch.digamma(strengths_tilde)[:, None]),
            dim=1,
        )
        loss = torch.mean(first + second)
        return loss


class EvidentialNIGNLLLoss(nn.Module):
    """Evidential normal inverse gamma negative log likelihood loss.

    Implementation is based on :cite:`aminiDeepEvidential2020`.
    """

    def __init__(self) -> None:
        """Intializes an instance of the EvidentialNIGNLLLoss class."""
        super().__init__()

    def forward(self, inputs: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential normal inverse gamma negative log likelihood loss.

        Args:
            inputs: dict[str, torch.Tensor] with keys 'gamma', 'nu', 'alpha', 'beta'
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        omega = 2 * inputs["beta"] * (1 + inputs["nu"])
        loss = (
            0.5 * torch.log(torch.pi / inputs["nu"])
            - inputs["alpha"] * torch.log(omega)
            + (inputs["alpha"] + 0.5) * torch.log((targets - inputs["gamma"]) ** 2 * inputs["nu"] + omega)
            + torch.lgamma(inputs["alpha"])
            - torch.lgamma(inputs["alpha"] + 0.5)
        ).mean()
        return loss


class EvidentialRegressionRegularization(nn.Module):
    """Implementation of the evidential regression regularization :cite:`aminiDeepEvidential2020`."""

    def __init__(self) -> None:
        """Initialize an instance of the evidential regression regularization class."""
        super().__init__()

    def forward(self, inputs: dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of the evidential regression regularization.

        Args:
            inputs: dict[str, torch.Tensor] with keys 'gamma', 'nu', 'alpha', 'beta'
            targets: torch.Tensor of size (n_instances,)

        Returns:
            loss: torch.Tensor, mean loss value

        """
        loss = (torch.abs(targets - inputs["gamma"]) * (2 * inputs["nu"] + inputs["alpha"])).mean()
        return loss


class IRDLoss(nn.Module):
    """Implementation of the Information-Robust Dirichlet Loss :cite:`tsiligkaridisInformationRobustDirichlet2019`.

    This loss function combines three terms:
    1. Calibration term (Lp loss) using beta function expectations
    2. Regularization term penalizing high alpha values for incorrect classes
    3. Adversarial entropy penalty for out-of-distribution robustness
    """

    def __init__(self, p: float = 2.0, lam: float = 0.15, gamma: float = 1.0) -> None:
        """Initialize an instance of the IRDLoss class.

        Args:
            p: float, Lp norm exponent for calibration loss (default: 2.0)
            lam: float, regularization weight (default: 0.15)
            gamma: float, entropy weight for adversarial robustness (default: 1.0)
        """
        super().__init__()
        self.p = p
        self.lam = lam
        self.gamma = gamma

    def _lp_fn(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Lp calibration loss (upper bound Fi).

        Args:
            alpha: Dirichlet concentration parameters, shape (B, K)
            y: One-hot encoded labels, shape (B, K)

        Returns:
            Scalar loss summed over batch
        """

        def log_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

        alpha0 = alpha.sum(dim=1, keepdim=True)
        alpha_c = (alpha * y).sum(dim=1, keepdim=True)

        log_e1 = log_b(alpha0 - alpha_c + self.p, alpha_c) - log_b(alpha0 - alpha_c, alpha_c)
        e1 = torch.exp(log_e1)  # Expected value of (1 - p_c)^p
        log_ep = log_b(alpha + self.p, alpha0 - alpha) - log_b(alpha, alpha0 - alpha)
        ep = torch.exp(log_ep) * (1 - y)  # Per-class expected values E[p_j^p]
        e_sum = e1 + ep.sum(dim=1, keepdim=True)  # E_sum = E1 + sum of incorrect class terms
        fi = torch.exp(torch.log(e_sum + 1e-8) / self.p).squeeze(1)

        return fi.sum()

    def _regularization_fn(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the regularization term using trigamma functions.

        Args:
            alpha: Dirichlet concentration parameters, shape (B, K)
            y: One-hot encoded labels, shape (B, K)

        Returns:
            Scalar regularization loss.
        """
        alpha_tilde = alpha * (1 - y) + y
        alpha_tilde_0 = torch.sum(alpha_tilde, dim=1, keepdim=True)
        trigamma_alpha = torch.polygamma(1, alpha_tilde)
        trigamma_alpha0 = torch.polygamma(1, alpha_tilde_0)
        diff_sq = (alpha_tilde - 1.0) ** 2  # (alpha_tilde - 1)^2 term, only for incorrect classes
        mask = 1 - y
        term = 0.5 * diff_sq * (trigamma_alpha - trigamma_alpha0) * mask

        return torch.sum(term)

    def _dirichlet_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute Dirichlet entropy for adversarial robustness.

        Args:
            alpha: Dirichlet concentration parameters, shape (B, K)

        Returns:
            Scalar entropy summed over batch
        """
        alpha0 = alpha.sum(dim=-1)
        log_b = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha0)
        term1 = log_b
        term2 = (alpha0 - alpha.size(-1)) * torch.digamma(alpha0)
        term3 = ((alpha - 1) * torch.digamma(alpha)).sum(dim=-1)
        entropy = term1 + term2 - term3

        return entropy.sum()

    def forward(
        self,
        alpha: torch.Tensor,
        y: torch.Tensor,
        adversarial_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the Information-Robust Dirichlet Loss.

        Args:
            alpha: torch.Tensor of shape (B, K) containing Dirichlet parameters
            y: torch.Tensor of shape (B,K)
            adversarial_alpha: torch.Tensor of shape (B_a, K)

        Returns:
            loss: torch.Tensor, scalar loss value
        """
        lp_term = self._lp_fn(alpha, y)
        reg_term = self._regularization_fn(alpha, y)
        entropy_term = self._dirichlet_entropy(adversarial_alpha) if adversarial_alpha is not None else 0.0

        return lp_term + self.lam * reg_term - self.gamma * entropy_term


class DirichletPriorNetworks(nn.Module):
    """Implementation of Dirichlet Prior Networks: cite:`malininPredictiveUncertaintyEstimation2018`.

    This class implements the Prior Networks framework with dual training on
    in-distribution (ID) and out-of-distribution (OOD) data using KL divergence
    between target and predicted Dirichlet distributions.
    """

    def __init__(
        self,
        alpha0_in: float = 100.0,
        alpha0_ood: float = 10.0,
        label_smoothing: float = 0.01,
        ce_weight: float = 0.1,
    ) -> None:
        """Initialize an instance of the DirichletPriorNetworks class.

        Args:
            alpha0_in: float, precision for in-distribution Dirichlet targets
                (higher = sharper distribution, default: 100.0)
            alpha0_ood: float, precision for OOD Dirichlet targets
                (lower = flatter distribution, default: 10.0)
            label_smoothing: float, label smoothing factor for numerical stability
                (default: 0.01)
            ce_weight: float, weight for cross-entropy regularization term
                (default: 0.1)
        """
        super().__init__()
        self.alpha0_in = alpha0_in
        self.alpha0_ood = alpha0_ood
        self.label_smoothing = label_smoothing
        self.ce_weight = ce_weight

    def _kl_dirichlet(
        self,
        alpha_target: torch.Tensor,
        alpha_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between target and predicted Dirichlet distributions.

        KL(Dir(alpha_target) || Dir(alpha_pred)) computed per batch element.

        Args:
            alpha_target: torch.Tensor of shape (B, K), target Dirichlet parameters
            alpha_pred: torch.Tensor of shape (B, K), predicted Dirichlet parameters

        Returns:
            torch.Tensor of shape (B,), KL divergence per sample
        """
        from torch.special import gammaln  # noqa: PLC0415

        alpha_p0 = alpha_target.sum(dim=-1, keepdim=True)
        alpha_q0 = alpha_pred.sum(dim=-1, keepdim=True)

        term1 = gammaln(alpha_p0) - gammaln(alpha_q0)
        term2 = (gammaln(alpha_pred) - gammaln(alpha_target)).sum(dim=-1, keepdim=True)
        term3 = ((alpha_target - alpha_pred) * (torch.digamma(alpha_target) - torch.digamma(alpha_p0))).sum(
            dim=-1,
            keepdim=True,
        )

        return (term1 + term2 + term3).squeeze(-1)

    def _make_in_domain_target_alpha(
        self,
        y: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Construct sharp Dirichlet targets for in-distribution samples.

        Args:
            y: torch.Tensor of shape (B,), class labels
            num_classes: int, number of classes

        Returns:
            torch.Tensor of shape (B, num_classes), target alpha for ID
        """
        batch_size = y.size(0)

        # Smoothed one-hot encoding
        mu = torch.full(
            (batch_size, num_classes),
            self.label_smoothing / (num_classes - 1),
            device=y.device,
        )
        mu[torch.arange(batch_size, device=y.device), y] = 1.0 - self.label_smoothing

        return mu * self.alpha0_in

    def _make_ood_target_alpha(
        self,
        batch_size: int,
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Construct flat Dirichlet targets for out-of-distribution samples.

        Args:
            batch_size: int, batch size
            num_classes: int, number of classes
            device: torch.device, device to create tensor on

        Returns:
            torch.Tensor of shape (batch_size, num_classes), target alpha for OOD
        """
        mu = torch.full(
            (batch_size, num_classes),
            1.0 / num_classes,
            device=device,
        )
        return mu * self.alpha0_ood

    def forward(
        self,
        alpha_pred: torch.Tensor,
        y: torch.Tensor,
        alpha_ood: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass computing the Prior Networks loss.

        Combines KL divergence for ID and OOD samples with optional
        cross-entropy regularization.

        Args:
            alpha_pred: torch.Tensor of shape (B, K), predicted Dirichlet
                parameters for in-distribution samples
            y: torch.Tensor of shape (B,), class labels for ID samples
            alpha_ood: torch.Tensor of shape (B_ood, K), optional predicted
                Dirichlet parameters for OOD samples

        Returns:
            torch.Tensor, scalar loss value
        """
        num_classes = alpha_pred.shape[1]
        device = alpha_pred.device

        # In-distribution loss: KL(target_sharp || predicted)
        alpha_target_in = self._make_in_domain_target_alpha(y, num_classes)
        kl_in = self._kl_dirichlet(alpha_target_in, alpha_pred).mean()

        # Cross-entropy term for classification stability
        alpha0_in = alpha_pred.sum(dim=-1, keepdim=True)
        probs_in = alpha_pred / alpha0_in
        ce_term = F.nll_loss(torch.log(probs_in + 1e-8), y)

        # Total loss with ID and optional OOD components
        total_loss = kl_in + self.ce_weight * ce_term

        # OOD loss: KL(target_flat || predicted)
        if alpha_ood is not None:
            alpha_target_ood = self._make_ood_target_alpha(
                alpha_ood.shape[0],
                num_classes,
                device,
            )
            kl_ood = self._kl_dirichlet(alpha_target_ood, alpha_ood).mean()
            total_loss = total_loss + kl_ood

        return total_loss
