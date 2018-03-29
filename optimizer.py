import numpy as np
from chainer.training import extension


class CosineAnnealing(extension.Extension):

    def __init__(self, attr, total_epoch, n_batch, eta_min=0.2 * 10 ** -2,
                 init=None, optimizer=None):
        self._attr = attr

        self.eta_max = init
        self.eta_min = eta_min
        self.total_epoch = total_epoch
        self.n_batch = n_batch

        self._init = init
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def calc_lr(self, i):
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
            (1 + np.cos(np.pi * float(i) / np.ceil(self.total_epoch * self.n_batch)))

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        value = self.calc_lr(self._t)

        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, np.ndarray):
            self._last_value = np.asscalar(self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
