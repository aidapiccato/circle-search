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
        self._n_items = self._map_circle._n_items
        self._n_colors = self._map_circle._n_colors
        self._prior = np.ones(self._n_items) / self._n_items
        non_target_color = np.arange(self._n_colors)
        non_target_color = non_target_color[non_target_color != self._target_color]
        self._n_s = len(np.where(self._map_circle._circle == non_target_color)[0])
        self._n_t = self._n_items - self._n_s

    # def _likelihood_source(self, theta, color):
    #     offsets = np.arange(1, self._n_items)

    def _likelihood_target(self, theta, color):
        offsets = np.arange(1, self._n_items)
        inside_source = np.maximum(0, self._n_s + offsets - self._n_items)
        outside_target = np.maximum(0, self._n_s - offsets)
        likelihood = (self._n_s - inside_source - outside_target) / self._n_s
        likelihood = np.hstack(([0], likelihood))
        likelihood = np.roll(likelihood, theta)
        return likelihood

    def _decision(self, posterior):
        return np.random.choice(np.flatnonzero(posterior == posterior.max()))

    def __call__(self, theta=None, color=None):
        if theta is None and color is None:
            return {'theta': np.random.randint(self._n_items, size=1)[0], 'posterior': self._prior}
        else:
            likelihood_target = self._likelihood_target(theta, color)
            # if self._target_color == 0:
            #     likelihood_target = 1 - likelihood_target
            posterior = likelihood_target * self._prior
            posterior = posterior/np.sum(posterior)
            self._prior = posterior
            return {'theta': self._decision(posterior), 'posterior': posterior}


