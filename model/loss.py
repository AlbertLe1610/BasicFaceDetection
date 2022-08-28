import torch


class RetinaNetBoxLoss:
    def __init__(self, beta=3):
        self._beta = beta

    def __call__(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = torch.abs(difference)
        squared_difference = difference ** 2
        loss = torch.where(
            torch.less(absolute_difference, self._beta),
            0.5 * squared_difference / self._beta,
            absolute_difference - 0.5 * self._beta,
        )
        return torch.mean(loss, dim=-1).sum()
