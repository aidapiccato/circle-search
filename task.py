import numpy as np
import copy

COLORS = 0
LOCS = 1
MIN_CIRCLE_SIZE = np.pi / 8


class CircleSearchAmbiguousFlat(object):
    def __init__(self, n_colors, n_attempts=np.inf, target_color=None, n_items=None):
        self._n_colors = n_colors
        self._n_attempts = n_attempts
        self._attempts = 0

        if target_color is None:
            self._target_color = np.random.randint(low=0, high=self._n_colors, size=1)
        else:
            self._target_color = target_color

        if n_items is None:
            self._n_items = np.random.randint(low=1, high=self._n_colors + 1)
        else:
            self._n_items = n_items

        self._angles = np.maximum(np.random.uniform(low=0, high=np.pi, size=self._n_items - 1), MIN_CIRCLE_SIZE)

        self._map_circle = self._generate_circle()
        self._occ_circle = copy.copy(self._map_circle)
        self._rot = np.random.uniform(low=0, high=np.pi)

        self._occ_circle[LOCS] = (self._occ_circle[LOCS] + self._rot) % (2 * np.pi)

    def _generate_circle(self):
        """
        Generates a circle with flat structured clusters.
        :return:
        """
        item_locs = np.zeros(shape=(1))

        item_locs = np.append(item_locs, self._angles)

        item_locs = np.cumsum(item_locs) % (2 * np.pi)

        rot = np.random.uniform(low=0, high=np.pi)

        item_locs = (item_locs + rot) % (2 * np.pi)

        # Determining color of items
        item_colors = []

        for item in range(self._n_items - 1):
            col = np.random.randint(low=0, high=self._n_colors)
            while col == self._target_color:
                col = np.random.randint(low=0, high=self._n_colors)
            item_colors.append(col)

        item_colors.append(self._target_color[0])

        np.random.shuffle(item_colors)

        item_colors = np.asarray(item_colors).astype(int)

        return np.vstack((item_colors, item_locs))

    def __call__(self, theta):
        color = self._occ_circle[COLORS][theta]
        self._attempts += 1
        trial_over = self._attempts > self._n_attempts or color == self._target_color
        resp = {'theta': theta, 'color': color}
        return trial_over, resp

    def model_args(self):
        return {'map_circle': self._map_circle, 'target_color': self._target_color}

    def task_dict(self):
        return {'map_circle': self._map_circle, 'occ_circle': self._occ_circle, 'target_color': self._target_color,
                'n_items': self._n_items, 'n_colors': self._n_colors, 'rot': self._rot, 'n_attempts':
                    self._n_attempts}

class CircleSearchAmbiguousClust(object):
    """
    A modified version of circle search task with a limited number of clustered circles. There is a single target
    item, all others are sampled with replacement
    """

    def __init__(self, n_colors, n_attempts=np.inf, target_color=None, n_items=None, n_clusters=None):
        self._n_colors = n_colors
        self._n_attempts = n_attempts
        self._attempts = 0

        if n_items is None:
            self._n_items = np.random.randint(low=n_colors, high=self._n_colors + 2)[0]
        else:
            self._n_items = n_items

        if target_color is None:
            self._target_color = np.random.randint(low=0, high=self._n_colors, size=1)
        else:
            self._target_color = target_color
        if n_clusters is None:
            self._n_clusters = np.random.randint(low=1, high=self._n_items + 1, size=1)[0]
        else:
            self._n_clusters = n_clusters

        self._map_circle = self._generate_circle()
        self._occ_circle = copy.copy(self._map_circle)
        self._rot = np.random.uniform(low=0, high=2 * np.pi)

        self._occ_circle[LOCS] = (self._occ_circle[LOCS] + self._rot) % (2 * np.pi)

        # TODO: Add functionality to pass in a ready-made circle

    def _generate_circle(self):
        """
        Generates a circle with hierarchically structured clusters.
        :return:
        """

        # Determines size of clusters (number of fruits)
        cluster_sizes = np.zeros(self._n_clusters)

        for clust in range(self._n_clusters - 1):
            cluster_sizes[clust] = np.random.randint(low=1, high=self._n_items - cluster_sizes.sum() -
                                                                 (self._n_clusters - clust - 1) + 1, size=1)[0]
        cluster_sizes[-1] = self._n_items - cluster_sizes.sum()

        np.random.shuffle(cluster_sizes)

        # Determine spacing of clusters
        item_locs = []
        #
        clust_size = 2 * np.pi / self._n_clusters
        for clust in range(self._n_clusters):
            clust_start = clust_size * clust
            clust_center = clust_size / 2 + clust_size * clust
            fruit_size = clust_size / cluster_sizes[clust]
            for item in range(int(cluster_sizes[clust])):
                item_locs.append(item * fruit_size / 2 + clust_start)

        # Determining color of items
        item_colors = []

        for item in range(self._n_items - 1):
            col = np.random.randint(low=0, high=self._n_colors)
            while col == self._target_color:
                col = np.random.randint(low=0, high=self._n_colors)
            item_colors.append(col)

        item_colors.append(self._target_color[0])

        np.random.shuffle(item_colors)

        item_colors = np.asarray(item_colors).astype(int)

        return np.vstack((item_colors, item_locs))

    def __call__(self, theta):
        color = self._occ_circle[COLORS][theta]
        self._attempts += 1
        trial_over = self._attempts > self._n_attempts or color == self._target_color
        resp = {'theta': theta, 'color': color}
        return trial_over, resp

    def model_args(self):
        return {'map_circle': self._map_circle, 'target_color': self._target_color}

    def task_dict(self):
        return {'map_circle': self._map_circle, 'occ_circle': self._occ_circle, 'target_color': self._target_color,
                'n_items': self._n_items, 'n_colors': self._n_colors, 'rot': self._rot, 'n_attempts':
                    self._n_attempts, 'n_clusters': self._n_clusters}


class CircleSearch(object):
    """Circle Search task.
    This task has a stimulus consisting of a colored circle, an occluded outer circle and a target color. The desired
    response is selection (saccade to) region on occluded outer ring matching target color.
    """

    def __init__(self, n_colors, n_items, n_attempts, n_regions=None, frac_target=None, map_circle=None,
                 occ_circle=None,
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

        if n_regions is None:
            self._n_regions = self._n_colors
        else:
            if self._n_colors == 2:
                n_regions = 2
            self._n_regions = n_regions

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
        self._n_target = len(np.flatnonzero(self._map_circle == self._target_color))
        self._model_args = {'map_circle': self._map_circle, 'target_color': self._target_color}

    def _generate_circle(self):
        n_target = np.maximum(1, np.minimum(np.ceil(self._frac_target * self._n_items), self._n_items - (
                self._n_regions - 1)))  #
        # size of target region is constrained such that there is enough space remaining in circle for all other colors
        # to have at least one sprite
        n_other_regions = self._n_regions - 1
        p_regions = np.ones(n_other_regions) / (n_other_regions)
        n_others = np.random.multinomial(self._n_items - n_target, p_regions)
        other_colors = list(set(np.arange(self._n_colors)) - {self._target_color[0]})
        region_colors = np.zeros(n_other_regions)
        if self._n_regions > self._n_colors:
            while len(np.unique(region_colors)) < self._n_colors - 1:
                region_colors[0] = np.random.choice(other_colors)
                for reg in range(1, self._n_regions - 1):
                    region_colors[reg] = np.random.choice(list(set(other_colors) - {region_colors[reg - 1]}))
        else:
            region_colors = other_colors
            np.random.shuffle(region_colors)

        region_colors = np.append(region_colors, self._target_color)
        region_sizes = np.append(n_others, n_target)

        circle = np.hstack(
            [np.repeat(region_colors[region], region_sizes[region]) for region in np.arange(self._n_regions)])

        circle = circle.astype(int)
        rot = np.random.randint(low=0, high=self._n_items - 1)

        return np.roll(circle, rot)

    def model_args(self):
        return self._model_args

    def task_dict(self):
        return {'map_circle': self._map_circle, 'occ_circle': self._occ_circle, 'target_color': self._target_color,
                'n_items': self._n_items, 'frac_target': self._frac_target, 'n_target': self._n_target,
                'n_colors': self._n_colors, 'rot': self._rot, 'n_attempts': self._n_attempts,
                'n_regions': self._n_regions}

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
    t = CircleSearchAmbiguousFlat(n_colors=4, n_items=4)
