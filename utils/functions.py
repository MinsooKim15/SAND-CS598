import torch
import numpy as np


def subsequent_mask(size: int) -> torch.Tensor:

    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("float32")
    mask = torch.from_numpy(mask) == 0
    return mask.float()


class ScheduledOptimizer:

    def __init__(self, optimizer, d_model: int, warm_up: int) -> None:
        self._optimizer = optimizer
        self.warm_up = warm_up
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> np.array:
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.warm_up, -1.5) * self.n_current_steps
        ])

    def get_lr(self):
        lr = self.init_lr * self._get_lr_scale()
        return lr

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return self._optimizer.state_dict()
