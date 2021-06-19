import numpy as np
import copy


class CircleSearch(object):
    """Circle Search task.
    This task has a stimulus consisting of a colored circle, an occluded outer circle and a target color. The desired
    response is selection (saccade to) region on occluded outer ring matching target color.
    """

    def __init__(self, n_colors, n_items, n_attempts, frac_target=None, map_circle=None, occ_circle=None,
                 target_color=None):
        """Constructor.
        Args:
            n_colors: Number of distinct colors present in circle
            n_items: Number of items present on circle
            n_attempts: Maximum number of attempts allowed during trial

        """
        self._n_colors = n_colors
        self._n_items = n_items
        self._n_attempts = n_attempts
        self._attempts = 0

        if target_color is None:
            self._target_color = np.random.randint(low=0, high=self._n_colors, size=1)
        else:
            self._target_color = target_color

        if frac_target is None:
            self._frac_target = np.random.uniform(low=0, high=1, size=1)  # selecting a random probability
        else:
            self._frac_target = frac_target

        if map_circle is None and occ_circle is None:  # not given map or search circles
            self._map_circle = self._generate_circle()
            self._occ_circle = copy.copy(self._map_circle)
            self._rot = np.random.randint(low=0, high=self._n_items - 1)
            self._occ_circle = np.roll(self._occ_circle, self._rot)
        else:
            self._map_circle = map_circle
            self._occ_circle = occ_circle
            rot = 0
            while not np.roll(self._map_circle, rot) == self._occ_circle:
                rot += 1
            self._rot = rot
        self._model_args = {'map_circle': self._map_circle, 'target_color': self._target_color}

    def _generate_circle(self):

        n_target = np.maximum(1, np.minimum(np.ceil(self._frac_target * self._n_items), self._n_items - (
                self._n_colors - 1)))  #
        # size of target region.
        # constrained such that there is enough space remaining in circle for all other colors to have at least one
        # sprite
        p_colors = np.ones(self._n_colors - 1) / (self._n_colors - 1)
        n_others = np.random.multinomial(self._n_items - n_target, p_colors)
        while len(np.flatnonzero(n_others == 0)) > 0:
            n_others = np.random.multinomial(self._n_items - n_target, p_colors)
        regions = np.zeros(self._n_colors)
        regions[np.arange(self._n_colors) != self._target_color] = n_others
        regions[self._target_color] = n_target
        colors = np.arange(self._n_colors)
        np.random.shuffle(colors)
        return np.hstack([np.repeat(color, regions[color]) for color in colors])

    def model_args(self):
        return self._model_args

    def task_dict(self):
        return {'map_circle': self._map_circle, 'occ_circle': self._occ_circle, 'target_color': self._target_color,
                'n_items': self._n_items, 'frac_target': self._frac_target,
                'n_colors': self._n_colors, 'rot': self._rot, 'n_attempts': self._n_attempts}

    def __call__(self, theta):
        color = self._occ_circle[theta]
        self._attempts += 1
        trial_over = self._attempts > self._n_attempts or color == self._target_color
        resp = {'theta': theta, 'color': color}
        return trial_over, resp


class Driver(object):
    def __init__(self, model_constructor, task_constructor, task_kwargs, n_trials):
        self._task_kwargs = task_kwargs
        self._model_constructor = model_constructor
        self._task_constructor = task_constructor
        self._n_trials = n_trials

    def __call__(self, log_dir=None):
        trials = []
        for trial in range(self._n_trials):
            task = self._task_constructor(**self._task_kwargs)
            trial_dict = task.task_dict()
            trial_dict.update({'n_attempts': 0, 'theta': [], 'posterior': []})
            model = self._model_constructor(**task.model_args())
            model_resp = model()
            theta = model_resp['theta']
            trial_over = False
            while not trial_over:
                trial_dict['theta'].append(theta)
                trial_dict['posterior'].append(model_resp['posterior'])
                trial_dict['n_attempts'] += 1
                trial_over, task_resp = task(theta)
                if not trial_over:
                    model_resp = model(**task_resp)
                    theta = model_resp['theta']
            trials.append(trial_dict)
        return trials


if __name__ == '__main__':
    t = CircleSearch(2, 16, np.inf, frac_target=0.1)
    print(t)