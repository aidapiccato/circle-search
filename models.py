"""Modules for template openmind project.

The classes in this file are just toy classes that do nothing interesting except
print and long things.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from python_utils.logging import scalar_writer as scalar_writer_lib


class IdealObserver():
    def __init__(self, map_circle, target_color):
        self._map_circle = map_circle
        self._target_color = target_color
        self._n_items = len(map_circle)
        self._n_colors = len(np.unique(map_circle))
        self._prior = np.ones(self._n_items) / self._n_items
        self._color_len = [len(np.where(self._map_circle == c)[0]) for c in range(self._n_colors)]
        non_target_color = np.arange(self._n_colors)
        non_target_color = non_target_color[non_target_color != self._target_color]
        self._n_s = len(np.where(self._map_circle == non_target_color)[0])
        self._n_t = self._n_items - self._n_s

    def _likelihood_target(self, theta, color):
        posterior = np.zeros(self._n_items)
        for rot in range(1, self._n_items):
            rot_circle = np.roll(self._map_circle, rot)
            if rot_circle[theta] == color:
                rot_circle_target = rot_circle == self._target_color
                posterior += rot_circle_target
        return posterior/self._n_items

    def _decision(self, posterior):
        choice = np.random.choice(np.flatnonzero(posterior == posterior.max()))
        return choice

    def __call__(self, theta=None, color=None):
        if theta is None and color is None:
            return {'theta': np.random.randint(self._n_items, size=1)[0], 'posterior': self._prior}
        else:
            likelihood_target = self._likelihood_target(theta, color)
            posterior = likelihood_target * self._prior
            posterior = posterior/np.sum(posterior)
            self._prior = posterior
            return {'theta': self._decision(posterior), 'posterior': posterior}


