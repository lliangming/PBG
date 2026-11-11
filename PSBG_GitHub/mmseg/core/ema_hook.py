import torch
from mmcv.runner import Hook


class EMAHook(Hook):
    """Exponential Moving Average of model weights for eval stability.

    Maintains a shadow copy of trainable parameters. On each iteration,
    swaps EMA weights into the model after the optimizer step so that
    downstream hooks (e.g. EvalHook) use the smoothed weights for
    evaluation. A companion restore hook swaps training weights back.

    Particularly useful when training with noisy gradient signals:
    OHEM, DiceLoss, small batch sizes, and heavy class imbalance.

    Args:
        decay (float): EMA decay rate. 0.999 → ~1000-step window.
        start_iter (int): Iteration to begin EMA updates.
    """

    PRIORITY = 'ABOVE_NORMAL'

    def __init__(self, decay=0.999, start_iter=0):
        self.decay = decay
        self.start_iter = start_iter
        self._ema_params = None
        self._training_params = None
        self._swapped = False

    def before_run(self, runner):
        model = self._unwrap(runner.model)
        self._ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._ema_params[name] = param.data.clone()

    def after_train_iter(self, runner):
        if runner.iter < self.start_iter:
            return

        model = self._unwrap(runner.model)

        # Update EMA shadow from current (post-step) training weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._ema_params[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

        # Swap EMA in so EvalHook (lower priority) uses smoothed weights
        self._swap_in(model)

    def _swap_in(self, model):
        if self._swapped:
            return
        self._training_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._training_params[name] = param.data.clone()
                param.data.copy_(self._ema_params[name])
        self._swapped = True

    def _swap_out(self, model):
        if not self._swapped:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self._training_params[name])
        self._training_params = None
        self._swapped = False

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, 'module') else model

    @property
    def ema_params(self):
        return self._ema_params


class EMARestoreHook(Hook):
    """Companion to EMAHook — swaps training weights back into the model.

    Registered at VERY_LOW priority so it runs after EvalHook (LOW).
    """

    PRIORITY = 'VERY_LOW'

    def __init__(self, ema_hook):
        self.ema_hook = ema_hook

    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        self.ema_hook._swap_out(model)
