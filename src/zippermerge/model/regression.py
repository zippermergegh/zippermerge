# Linear regression trained with gradient descent in PyTorch.

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def _to_1d_tensor(data) -> torch.Tensor:
    arr = np.asarray(data, dtype=np.float64).reshape(-1)
    return torch.from_numpy(arr)


class LinearRegression(nn.Module):
    """Simple linear model  y = w0 + w1 * x  fitted by gradient descent (SGD).

    learning_rate : float
        Step size for the SGD optimizer.
    epochs : int
        Number of passes over the training data.
    """

    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if int(epochs) <= 0:
            raise ValueError("epochs must be a positive integer.")

        self.learning_rate = learning_rate
        self.epochs = int(epochs)

        # Trainable parameters (intercept w0 and slope w1)
        self.w0 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.w1 = nn.Parameter(torch.zeros(1, dtype=torch.float64))

        # Optimizer and loss function
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Training history
        self.w0_history: list[float] = []
        self.w1_history: list[float] = []
        self.loss_history: list[float] = []

        # Data kept after fit() so analysis_plot() can use it
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self.r2_test = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear model: y = w0 + w1 * x."""
        return self.w0 + self.w1 * x

    def fit(self, x_train, y_train, x_test, y_test) -> float:
        """Train on (x_train, y_train), then compute and print R^2 on the test data.

        Returns the test R^2.
        """
        x_tr, y_tr = _to_1d_tensor(x_train), _to_1d_tensor(y_train)
        x_te, y_te = _to_1d_tensor(x_test), _to_1d_tensor(y_test)

        with torch.no_grad():
            self.w0.zero_()
            self.w1.zero_()
        self.w0_history.clear()
        self.w1_history.clear()
        self.loss_history.clear()

        self.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            y_pred = self.forward(x_tr)
            loss = self.loss_fn(y_pred, y_tr)

            # Record the parameters used for this loss value
            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

            loss.backward()
            self.optimizer.step()

        # Evaluate on the test set
        self.eval()
        with torch.no_grad():
            y_hat = self.forward(x_te)
            ss_res = torch.sum((y_te - y_hat) ** 2)
            ss_tot = torch.sum((y_te - torch.mean(y_te)) ** 2)
            self.r2_test = (1.0 - ss_res / ss_tot).item()

        self._x_train, self._y_train = x_tr.numpy(), y_tr.numpy()
        self._x_test, self._y_test = x_te.numpy(), y_te.numpy()

        print(f"Learned parameters: w0 = {self.w0.item():.4f}, w1 = {self.w1.item():.4f}")
        print(f"R^2 on test data: {self.r2_test:.4f}")
        return self.r2_test

    def predict(self, x_new) -> np.ndarray:
        """Predict AnnualProduction for new BCR value(s)."""
        x = _to_1d_tensor(x_new)
        self.eval()
        with torch.no_grad():
            return self.forward(x).numpy()

    def analysis_plot(self, save_path: str | None = None):
        """Plot data + fitted line, and w0, w1, loss versus epoch on one figure."""
        if not self.loss_history:
            raise RuntimeError("Call fit() before analysis_plot().")

        epochs = np.arange(1, len(self.loss_history) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Original data and fitted regression line
        ax = axes[0, 0]
        ax.scatter(self._x_train, self._y_train, s=18, alpha=0.6, label="Train data")
        ax.scatter(self._x_test, self._y_test, s=18, alpha=0.6, color="orange",
                   label="Test data")
        all_x = np.concatenate([self._x_train, self._x_test])
        x_line = np.linspace(all_x.min(), all_x.max(), 200)
        ax.plot(x_line, self.predict(x_line), color="red", lw=2,
                label=f"Fit: y = {self.w0.item():.1f} + {self.w1.item():.1f}x")
        ax.set_xlabel("BCR")
        ax.set_ylabel("AnnualProduction")
        ax.set_title("Data and fitted regression line")
        ax.legend()

        # w0 over training
        axes[0, 1].plot(epochs, self.w0_history, color="tab:green")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("w0")
        axes[0, 1].set_title("Intercept w0 during training")

        # w1 over training
        axes[1, 0].plot(epochs, self.w1_history, color="tab:purple")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("w1")
        axes[1, 0].set_title("Slope w1 during training")

        # Loss over training
        axes[1, 1].plot(epochs, self.loss_history, color="tab:red")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("MSE loss (log scale)")
        axes[1, 1].set_title("Training loss")

        fig.suptitle("Linear regression via gradient descent: training analysis")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()
        return fig