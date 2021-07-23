"""Task with predators chasing agent in open arena.
The predators (red circles) chase the agent. The predators bouce off the arena
boundaries, while the agent cannot exit but does not bounce (i.e. it has
inelastic collisions with the boundaries). Trials only terminate when the agent
is caught by a predator. The subject controls the agent with a joystick.
This task also contains an auto-curriculum: When the subject does well (evades
the predators for a long time before being caught), the predators' masses are
decreased, thereby increasing the predators' speeds. Conversely, when the
subject does poorly (gets caught quickly), the predators' masses are increased,
thereby decreasing the predators' speeds.
"""

import collections
import numpy as np

from moog import shapes
from moog import action_spaces
from moog import game_rules
from moog import observers
from moog import physics as physics_lib
from moog import tasks
from moog import sprite

from moog_configs.rules import ModifyWhenContacting
from moog_configs.rules import ModifyMetaStateOnContact
from moog_configs.rules import RecenterAgent
from moog_configs.rules import ModifyMetaStateDelay
from task import CircleSearchAmbiguousClust

_RED = [0, 1.0, 1.0]
_GREEN = [0.5, 1.0, 1.0]
_BLUE = [0.7, 1.0, 1.0]
_YELLOW = [0.18, .98, 1.0]
_PURPLE = [0.62, 0.63, 1]
_PINK = [0.9, 0.376, 1]
_WHITE = [1.0, 0, 1.0]
_TURQUOISE = [0.27, 1, .8]
_COLORS = np.asarray([_RED, _GREEN, _BLUE, _YELLOW, _PURPLE, _PINK, _WHITE, _TURQUOISE])
_GRAY = [0, 0, 0.7]
_N_CIRCLES = 18

_META_STATE_CONTACTING = 1
_META_STATE_SEARCH = 0
_META_STATE_SCREEN = -10


def _make_opaque(s):
    s.opacity = 255


def _make_transparent(s):
    s.opacity = 0


def _set_color(s, color):
    s.c0 = color[0]
    s.c1 = color[1]
    s.c2 = color[2]


def _center_sprite(s):
    s.position = (0.5, 0.5)


def _color_fruit(s):
    _set_color(s, s.metadata['color'])


def _hide_fruit(s):
    _set_color(s, _GRAY)


class StateInitialization:
    """State initialization class to dynamically adapt predator mass.

    This is essentially an auto-curriculum: When the subject does well (evades
    the predators for a long time before being caught), the predators' masses
    are decreased, thereby increasing the predators' speeds. Conversely, when
    the subject does poorly (gets caught quickly), the predators' masses are
    increased, thereby decreasing the predators' speeds.
    """

    def __init__(self, max_attempts=np.inf, n_colors=None):
        """Constructor.
        This class uses the meta-state to keep track of the number of steps
        before the agent is caught. See the game rules section near the bottom
        of this file for the counter incrementer.
        Args:
            step_scaling_factor: Float. Fractional decrease of predator mass
                after a trial longer than threshold_trial_len. Also used as
                fractional increase of predator mass after a trial shorter than
                threshold_trial_len. Should be small and positive.
            threshold_trial_len: Length of a trial above which the predator
                mass is decreased and below which the predator mass is
                increased.
        """

        # Agent
        if n_colors is None:
            n_colors = [3, 4]

        def _agent_generator(n_attempts):
            return sprite.Sprite(x=0.5, y=0.5, shape='circle', scale=0.03, c0=0.33, c1=1., c2=0.66, metadata={
                'n_attempts': n_attempts})

        self._agent_generator = _agent_generator

        # Fruits

        def _fruit_generator(fruit_colors, fruits_rad, target_color, r=0.35):
            n_circles = len(fruit_colors)
            fruit_factors = {'shape': 'circle', 'scale': 0.05}
            fruits_grid_x = r * np.cos(fruits_rad) + 0.5
            fruits_grid_y = r * np.sin(fruits_rad) + 0.5
            fruit_theta = np.arange(n_circles)
            fruits_props = zip(fruits_grid_x, fruits_grid_y, fruit_colors, fruit_theta)

            fruit_sprites = [
                sprite.Sprite(x=x, y=y, c0=c0, c1=c1, c2=c2,
                              metadata={'theta': theta, 'color': (c0, c1, c2), 'target': (list((c0, c1,
                                                                                                c2)) ==
                                                                                          list(target_color))},
                              **fruit_factors)
                for (x, y, (c0, c1, c2), theta) in fruits_props
            ]
            return fruit_sprites

        def _cue_generator(target_color):
            cue = sprite.Sprite(x=0.5, y=0.5, shape='circle', scale=0.1, opacity=0)
            _set_color(cue, target_color)
            return cue

        self._n_colors = n_colors

        self._cue_generator = _cue_generator

        self._fruit_generator = _fruit_generator

        self._meta_state = None

        self._agent = None

        self._target = None

        self._task = None

        self._n_items = None

        self._max_attempts = max_attempts

    def state_initializer(self):
        """State initializer method to be fed to environment."""

        # Generating number of colors and number of items
        n_colors = np.random.choice([3, 4], size=1)

        # Generating task object
        task = CircleSearchAmbiguousClust(n_colors=n_colors, n_attempts=self._max_attempts).task_dict()

        self._task = task
        agent = self._agent_generator(self._max_attempts)
        colors_shuffled = _COLORS
        np.random.shuffle(colors_shuffled)
        target_color = colors_shuffled[int(task['target_color'])]

        # Generating map fruits
        fruit_colors_map = [colors_shuffled[int(color)] for color in task['map_circle'][0]]
        fruits_rad_map = task['map_circle'][1]
        map_fruits = self._fruit_generator(fruit_colors_map, fruits_rad_map, target_color, r=0.15)

        # Generating fruits
        fruit_colors_occ = [colors_shuffled[int(color)] for color in task['occ_circle'][0]]
        fruits_rad_occ = task['occ_circle'][1]
        fruits = self._fruit_generator(fruit_colors_occ, fruits_rad_occ, target_color, r=0.25)

        # Generating cue
        cue = self._cue_generator(target_color)

        # Generating circle
        annulus_verts_occ = shapes.annulus_vertices(0.245, 0.255)
        annulus_verts_map = shapes.annulus_vertices(0.145, 0.155)
        annulus_occ = sprite.Sprite(x=0.5, y=0.5, shape=annulus_verts_occ, scale=1., c0=0, c1=0, c2=.7)
        annulus_map = sprite.Sprite(x=0.5, y=0.5, shape=annulus_verts_map, scale=1., c0=0, c1=0, c2=.7)
        state = collections.OrderedDict([
            ('annuli', (annulus_occ, annulus_map)),
            ('cue', (cue,)),
            ('agent', (agent,)),
            ('fruits', fruits),
            ('map_fruits', map_fruits)
        ])

        return state

    def meta_state_initializer(self):
        """Meta-state initializer method to be fed to environment."""
        self._meta_state = {'phase': '', 'contacted_fruits': [], 'trial': self._task}
        return self._meta_state


def get_config(level, max_attempts=-1):
    """Get config dictionary of kwargs for environment constructor.

    Args:
        level: Int. Number of circles.
    """

    ############################################################################
    # Sprite initialization
    ############################################################################

    state_initialization = StateInitialization(
        max_attempts=np.inf
    )

    ############################################################################
    # Physics
    ############################################################################

    agent_friction_force = physics_lib.Drag(coeff_friction=0.25)

    forces = (
        (agent_friction_force, 'agent'),
    )

    physics = physics_lib.Physics(*forces, updates_per_env_step=10)

    ############################################################################
    # Task
    ############################################################################

    def _eq_sprite_color(s, color):
        return list((s.c0, s.c1, s.c2)) == list(color)

    def _fruit_reward_fn(_, fruit_sprite):
        hit_target = fruit_sprite.metadata['target']
        return 1 * hit_target + -1 * (not hit_target)

    contact_task = tasks.ContactReward(
        reward_fn=_fruit_reward_fn, layers_0='agent', layers_1='fruits')

    def _should_reset(state, meta_state):
        for s in state['fruits']:
            if s.overlaps_sprite(state['agent'][0]) and s.metadata['target']:
                return True
        if len(meta_state['contacted_fruits']) >= state['agent'][0].metadata['n_attempts']:
            return True
        return False

    reset_task = tasks.Reset(condition=_should_reset, steps_after_condition=10)

    task = tasks.CompositeTask(contact_task, reset_task)
    ############################################################################
    # Action space
    ############################################################################
    #
    # action_space = action_spaces.Joystick(
    #     scaling_factor=0.01, action_layers='agent')

    action_space = action_spaces.Grid(action_layers='agent', scaling_factor=0.01)

    # action_space = action_spaces.SetPosition(action_layers='agent', inertia=0.05)

    # Observer
    ############################################################################

    observer = observers.PILRenderer(
        image_size=(64, 64), anti_aliasing=1, color_to_rgb='hsv_to_rgb')

    ############################################################################
    # Game rules
    ############################################################################

    def _update_contacted_fruits(meta_state, contacted_fruits):
        # if meta_state['phase'] != 'contacting':
        thetas = [fruit.metadata['theta'] for fruit in contacted_fruits]
        meta_state_contacted = set(meta_state['contacted_fruits'])
        meta_state['contacted_fruits'] = list(meta_state_contacted.union(set(thetas)))

    update_rule = ModifyMetaStateOnContact(layers_0='agent', layers_1='fruits', modifier=_update_contacted_fruits)

    def update_meta_state_contact(meta_state, contacts):
        meta_state['phase'] = _META_STATE_CONTACTING

    update_meta_state_contact = ModifyMetaStateOnContact(layers_0='agent', layers_1='fruits',
                                                         modifier=update_meta_state_contact)
    contact_rule = ModifyWhenContacting(
        'fruits', 'agent', on_contact=_color_fruit, off_contact=_hide_fruit)

    center_rule = game_rules.ModifySprites('agent', _center_sprite)
    center_rule = RecenterAgent(delay=11, rules=(center_rule,))

    def update_meta_state_search(meta_state):
        meta_state['phase'] = _META_STATE_SEARCH

    update_meta_state_search = game_rules.ModifyMetaState(modifier=update_meta_state_search)
    update_meta_state_search_delay = ModifyMetaStateDelay(delay=11, rules=(update_meta_state_search,))

    # What should phase sequence look like
    # screen_phase (target cue, all else hidden) -> search phase (all visible)
    # during the search phase -> on contact there is some delay, then the
    show_cue = game_rules.ModifySprites('cue', _make_opaque)
    hide_annulus = game_rules.ModifySprites('annuli', _make_transparent)
    show_annulus = game_rules.ModifySprites('annuli', _make_opaque)
    hide_fruits = game_rules.ModifySprites('fruits', _make_transparent)
    hide_agent = game_rules.ModifySprites('agent', _make_transparent)
    hide_map_fruits = game_rules.ModifySprites('map_fruits', _make_transparent)

    def update_meta_state_screen(meta_state):
        meta_state['phase'] = _META_STATE_SCREEN

    update_meta_state_screen = game_rules.ModifyMetaState(modifier=update_meta_state_screen)
    screen_phase = game_rules.Phase(one_time_rules=(show_cue, hide_fruits, hide_agent, hide_map_fruits,
                                                    update_meta_state_screen, hide_annulus), duration=40,
                                    name='screen')

    # Search phase
    hide_cue = game_rules.ModifySprites('cue', _make_transparent)
    show_fruits = game_rules.ModifySprites('fruits', _make_opaque)
    show_agent = game_rules.ModifySprites('agent', _make_opaque)
    show_map_fruits = game_rules.ModifySprites('map_fruits', _make_opaque)
    occlude_fruits = game_rules.ModifySprites('fruits', _hide_fruit)

    continual_rules = (contact_rule, update_rule, center_rule, update_meta_state_contact,
                       update_meta_state_search_delay)
    search_phase = game_rules.Phase(one_time_rules=(hide_cue, show_annulus, show_fruits, show_agent, show_map_fruits,
                                                    occlude_fruits,
                                                    update_meta_state_search),
                                    continual_rules=continual_rules,
                                    name='search')
    phase_sequence = game_rules.PhaseSequence(screen_phase, search_phase)
    ############################################################################
    # Final config
    ############################################################################

    config = {
        'state_initializer': state_initialization.state_initializer,
        'physics': physics,
        'game_rules': (phase_sequence,),
        'task': task,
        'action_space': action_space,
        'observers': {'image': observer},
        'meta_state_initializer': state_initialization.meta_state_initializer,
    }
    return config
