from __future__ import annotations

from typing import Any

import pytest
from torch.utils.data import DataLoader, TensorDataset

from probly.layers.evidential.torch import BatchedRadialFlowDensity
from probly.losses.evidential.torch import (
    der_loss,
    evidential_ce_loss,
    loss_ird,
    natpn_loss,
    pn_loss,
    postnet_loss,
    rpn_loss,
)
from probly.train.evidential.torch import unified_evidential_train

torch = pytest.importorskip("torch")
from torch import Tensor, nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402


class DummyEDL(nn.Module):
    def __init__(self, in_dim: int = 4, num_classes: int = 3) -> None:
        """Dummy class for EDL."""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class DummyNatPN(nn.Module):
    def __init__(self, in_dim: int = 4, num_classes: int = 3) -> None:
        """Dummy class for NatPN."""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Any, Any]:
        alpha = F.softplus(self.fc(x)) + 1.0
        return alpha, None, None


class DummyIRD(nn.Module):
    def __init__(self, in_dim: int = 4, num_classes: int = 3) -> None:
        """Dummy class for IRD."""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(self.fc(x)) + 1e-4


class DummyDER(nn.Module):
    def __init__(self, in_dim: int = 4) -> None:
        """Dummy class for DER."""
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        out = self.fc(x)
        mu = out
        kappa = F.softplus(out) + 1.0
        alpha = F.softplus(out) + 1.0
        beta = F.softplus(out) + 1.0
        return mu, kappa, alpha, beta


class DummyPostNet(nn.Module):
    def __init__(self, in_dim: int = 4, latent_dim: int = 8) -> None:
        """Dummy class for PostNet."""
        super().__init__()
        self.fc = nn.Linear(in_dim, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class DummyPN(nn.Module):
    def __init__(self, in_dim: int = 4, num_classes: int = 10) -> None:
        """Dummy class for Prior Network."""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(self.fc(x)) + 1.0


class DummyRPN(nn.Module):
    def __init__(self, in_dim: int = 4) -> None:
        """Dummy class for Regression Prior Network."""
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        out = self.fc(x)
        mu = out
        kappa = F.softplus(out) + 1.0
        alpha = F.softplus(out) + 1.0
        beta = F.softplus(out) + 1.0
        return mu, kappa, alpha, beta


def make_dataloader(batch_size: int = 4) -> DataLoader[tuple[Tensor, Tensor]]:
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


def make_dataloader_pn(num_classes: int = 10) -> DataLoader[tuple[Tensor, Tensor]]:
    x = torch.randn(num_classes, 4)
    y = torch.arange(num_classes)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=num_classes)


def make_oodloader(batch_size: int = 4) -> DataLoader[tuple[Tensor, Tensor]]:
    x = torch.randn(8, 4) * 3.0 + 5.0
    y = torch.zeros(8)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


@pytest.mark.parametrize(
    ("mode", "model_cls", "loss_fn"),
    [
        ("EDL", DummyEDL, evidential_ce_loss),
        ("NatPostNet", DummyNatPN, natpn_loss),
        ("IRD", DummyIRD, loss_ird),
        ("DER", DummyDER, der_loss),
    ],
)
def test_train_simple_modes(mode, model_cls, loss_fn) -> None:
    model = model_cls()
    dataloader = make_dataloader()

    unified_evidential_train(
        mode=mode,
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        epochs=1,
        lr=1e-2,
        device="cpu",
    )


def test_train_postnet() -> None:
    num_classes = 3
    latent_dim = 8

    model = DummyPostNet(in_dim=4, latent_dim=latent_dim)
    dataloader = make_dataloader()

    flow = BatchedRadialFlowDensity(
        num_classes=num_classes,
        dim=latent_dim,
        flow_length=2,
    )

    class_count = torch.ones(num_classes)

    unified_evidential_train(
        mode="PostNet",
        model=model,
        dataloader=dataloader,
        loss_fn=postnet_loss,
        flow=flow,
        class_count=class_count,
        epochs=1,
        lr=1e-3,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("mode", "model_cls", "loss_fn", "num_classes"),
    [
        ("PrNet", DummyPN, pn_loss, 10),
        ("RPN", DummyRPN, rpn_loss, 1),
    ],
)
def test_train_ood_modes(mode, model_cls, loss_fn, num_classes) -> None:
    model = model_cls()
    dataloader = make_dataloader_pn(num_classes=num_classes)
    oodloader = make_oodloader()

    unified_evidential_train(
        mode=mode,
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        oodloader=oodloader,
        epochs=1,
        lr=1e-3,
    )
