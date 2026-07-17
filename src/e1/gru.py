"""
Modelo GRU para E1 (Estrategia Conservadora).

Arquitectura según especificación:
- Input: (sequence_length=360, features=F)
- GRU 1: 128 units, return_sequences=True, recurrent_dropout=0.1
- Dropout: 0.2
- GRU 2: 64 units, return_sequences=False, recurrent_dropout=0.1
- Dropout: 0.2
- Dense: 32 units, activation='relu'
- Output: 1 (retorno a H días)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'torch'. Install with: pip install torch"
        ) from exc


@dataclass
class TrainResult:
    best_val_loss: float
    epochs_ran: int


class GRURegressor:
    """GRU para regresión de retornos (E1)."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = None,
        dropout: float = 0.2,
        dense_units: int = 32,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        torch = _require_torch()

        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.torch = torch
        self.device = torch.device(device)

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                # GRU layers
                # Nota: dropout en GRU solo funciona con num_layers > 1
                # Usamos torch.nn.Dropout manualmente después de cada capa
                self.gru1 = torch.nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_sizes[0],
                    batch_first=True,
                    dropout=0.0,  # Siempre 0, usamos Dropout manual
                )
                self.dropout1 = torch.nn.Dropout(dropout)

                if len(hidden_sizes) > 1:
                    self.gru2 = torch.nn.GRU(
                        input_size=hidden_sizes[0],
                        hidden_size=hidden_sizes[1],
                        batch_first=True,
                        dropout=0.0,  # Siempre 0, usamos Dropout manual
                    )
                    self.dropout2 = torch.nn.Dropout(dropout)
                    final_hidden = hidden_sizes[1]
                else:
                    self.gru2 = None
                    self.dropout2 = None
                    final_hidden = hidden_sizes[0]

                # Dense layers
                self.dense = torch.nn.Linear(final_hidden, dense_units)
                self.relu = torch.nn.ReLU()
                self.output = torch.nn.Linear(dense_units, 1)

            def forward(self, x):
                # GRU 1
                out, _ = self.gru1(x)
                out = self.dropout1(out)

                # GRU 2 (si existe)
                if self.gru2 is not None:
                    out, _ = self.gru2(out)
                    out = self.dropout2(out)

                # Tomar último timestep
                out = out[:, -1, :]

                # Dense layers
                out = self.dense(out)
                out = self.relu(out)
                out = self.output(out)

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
        batch_size: int = 64,
        max_epochs: int = 200,
        early_stopping_patience: int = 15,
        loss: str = "huber",
        huber_delta: float = 1.0,
        verbose: bool = True,
    ) -> TrainResult:
        """Entrena el modelo GRU."""
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
            # Huber (SmoothL1)
            criterion = torch.nn.SmoothL1Loss(beta=huber_delta)

        # AdamW según especificación
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        ran = 0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb in loader:
                optim.zero_grad(set_to_none=True)
                pred = self.model(xb)
                l = criterion(pred, yb)
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()
                train_losses.append(float(l.item()))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xva)
                val_loss = float(criterion(val_pred, yva).item())

            ran = epoch
            pct_complete = 100.0 * epoch / max_epochs
            is_best = val_loss < best_val

            if verbose and (epoch % 5 == 0 or epoch == 1 or is_best or epoch == max_epochs):
                status = "✓" if is_best else " "
                train_loss_avg = float(np.mean(train_losses))
                print(
                    f"    Epoch {epoch:3d}/{max_epochs} ({pct_complete:5.1f}%): "
                    f"train_loss={train_loss_avg:.6f} val_loss={val_loss:.6f} {status}"
                )

            if is_best:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
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
        """Predice retornos."""
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
