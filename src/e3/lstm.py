from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'torch'. Install with: pip install torch"
        ) from exc


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_ran: int


class LSTMRegressor:
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dense_units: Optional[int] = 32,
        dropout: float = 0.2,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        torch = _require_torch()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.torch = torch
        self.device = torch.device(device)

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                if dense_units:
                    self.head = torch.nn.Sequential(
                        torch.nn.Linear(hidden_size, dense_units),
                        torch.nn.ReLU(),
                        torch.nn.Linear(dense_units, 1),
                    )
                else:
                    self.head = torch.nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.head(out)
                return out.squeeze(-1)

        self.model = _Net().to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        clipnorm: float = 1.0,
        loss: str = "huber",
        huber_delta: float = 1.0,
        verbose: bool = True,
    ) -> TrainResult:
        torch = self.torch

        Xtr = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        ytr = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        Xva = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        yva = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        ds = torch.utils.data.TensorDataset(Xtr, ytr)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

        if loss == "mse":
            criterion = torch.nn.MSELoss()
        else:
            # Huber (SmoothL1) with beta ~= delta
            criterion = torch.nn.SmoothL1Loss(beta=huber_delta)

        optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        ran = 0

        if verbose:
            print(f"    Training LSTM: {len(X_train)} train samples, {len(X_val)} val samples")

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb in loader:
                optim.zero_grad(set_to_none=True)
                pred = self.model(xb)
                l = criterion(pred, yb)
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clipnorm)
                optim.step()
                train_losses.append(float(l.item()))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xva)
                val_loss = float(criterion(val_pred, yva).item())

            ran = epoch

            # Print progress every 5 epochs or on improvement
            train_loss_avg = float(np.mean(train_losses))
            is_best = val_loss < best_val
            pct_complete = 100.0 * epoch / max_epochs

            if verbose and (epoch % 5 == 0 or epoch == 1 or is_best or epoch == max_epochs):
                status = "✓" if is_best else " "
                print(f"    Epoch {epoch:3d}/{max_epochs} ({pct_complete:5.1f}%): train_loss={train_loss_avg:.6f} val_loss={val_loss:.6f} {status}")

            if is_best:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stopping_patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch} ({pct_complete:5.1f}%, patience={early_stopping_patience})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if verbose:
            print(f"    Training complete: {ran} epochs, best_val_loss={best_val:.6f}")

        return TrainResult(best_val_loss=best_val, epochs_ran=ran)

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        torch = self.torch
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(Xt), batch_size):
                chunk = Xt[i : i + batch_size]
                out = self.model(chunk).detach().cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0)
