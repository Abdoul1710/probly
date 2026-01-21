from __future__ import annotations

import pytest
from torch.utils.data import DataLoader, TensorDataset

from probly.losses.evidential.torch import evidential_ce_loss, loss_ird, natpn_loss
from probly.train.evidential.torch import unified_evidential_train

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


class DummyEDL(nn.Module):
    def __init__(self, in_dim=4, num_classes=3) -> None:
        """Dummy class for EDL."""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class DummyNatPN(nn.Module):
    def __init__(self, in_dim=4, num_classes=3) -> None:
        """Dummy class for NatPN."""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        alpha = torch.nn.functional.softplus(self.fc(x)) + 1.0
        return alpha, None, None


class DummyIRD(nn.Module):
    def __init__(self, in_dim=4, num_classes=3) -> None:
        """Dummy class for IRD."""
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return torch.nn.functional.softplus(self.fc(x)) + 1e-4


def make_dataloader(batch_size=4):
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)

@pytest.mark.parametrize(
        ("mode", "model_cls", "loss_fn"),
        [
            ("EDL", DummyEDL, evidential_ce_loss),
            ("NatPostNet", DummyNatPN, natpn_loss),
            ("IRD", DummyIRD, loss_ird),
        ],
)

def test_train_simple_modes(mode, model_cls, loss_fn):
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
