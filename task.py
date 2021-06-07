import numpy as np
import copy
from matplotlib import cm


class Circle(object):
    """Circle object class
    To be used in CircleSearch task. Consists of neighborhoods of particular colors arranged in circular formation
    """

    def __init__(self, n_colors, n_items):
        self._n_colors = n_colors
        self._n_items = n_items
        regions = np.random.multinomial(self._n_items, np.ones(self._n_colors) / self._n_colors, size=1)[0]
        while(len(np.flatnonzero(regions == 0)) > 0):
            regions = np.random.multinomial(self._n_items, np.ones(self._n_colors) / self._n_colors, size=1)[0]
        colors = np.arange(self._n_colors)
        np.random.shuffle(colors)
        self._circle = np.hstack([np.repeat(color, regions[color]) for color in colors])

    def __getitem__(self, theta):
        return self._circle[theta]

    def plot(self, ax):
        cmap = cm.get_cmap('tab10', 6)
        ax.imshow(self._circle[:, np.newaxis].T / self._n_colors, cmap=cmap, interpolation='nearest', aspect='auto')
        ax.set_xlabel('$\\theta$')
        ax.set_yticks([])

    def rotate(self, theta):
        self._circle = np.roll(self._circle, theta)

    @property
    def circle(self):
        return self._circle


class CircleSearch(object):
    """Circle Search task.
    This task has a stimulus consisting of a colored circle, an occluded outer circle and a target color. The desired
    response is selection (saccade to) region on occluded outer ring matching target color.
    """

    def __init__(self, n_colors, n_items, n_attempts):
        """Constructor.
        Args:
            n_colors: Number of distinct colors present in circle
            n_items: Number of items present on circle
            n_attempts: Maximum number of attempts per trial
            p_sizes: Probability distribution used as input to circle search task
        """
        self._n_colors = n_colors
        self._n_items = n_items
        self._n_attempts = n_attempts
        self._attempts = 0
        self._map_circle = Circle(n_colors=self._n_colors, n_items=self._n_items)
        self._occ_circle = copy.copy(self._map_circle)
        self._occ_circle.rotate(np.random.randint(low=0, high=self._n_items - 1))
        self._target_color = np.random.choice(self._n_colors, size=1)[0]
        self._model_args = {'map_circle': self._map_circle, 'target_color': self._target_color}

    def model_args(self):
        return self._model_args

    def task_dict(self):
        return {'map_circle': self._map_circle, 'occ_circle': self._occ_circle, 'target_color': self._target_color,
                'n_items': self._n_items, 'n_target': len(np.where(self._occ_circle.circle == self._target_color)[0]),
                'n_colors': self._n_colors}

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
            # print(f'NEW TRIAL {trial}')
            while not trial_over:
                trial_dict['theta'].append(theta)
                trial_dict['posterior'].append(model_resp['posterior'])
                trial_dict['n_attempts'] += 1
                trial_over, task_resp = task(theta)
                # print('circle')
                # print(trial_dict['occ_circle']._circle)
                # print('attempts')
                # print(trial_dict['n_attempts'])
                # print('target_color')
                # print(trial_dict['target_color'])
                # print('color')
                # print(task_resp['color'])
                # print('theta')
                # print(theta)
                if not trial_over:
                    model_resp = model(**task_resp)
                    theta = model_resp['theta']
            trials.append(trial_dict)
        return trials
