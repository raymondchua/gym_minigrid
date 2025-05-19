from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=None,
        agent_start_dir=None,
        goal_pos=None,
        max_steps=None,
        show_goal: bool = True,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self.size_without_walls = size - 2
        self.show_goal = show_goal

        if max_steps is not None:
            self.max_steps = max_steps
        else:
            self.max_steps = (
                4 * (self.size_without_walls - 2) * (self.size_without_walls - 2)
            )

        super().__init__(
            grid_size=size,
            max_steps=self.max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            goal_pos=self.goal_pos,
            show_goal=self.show_goal,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        if self.goal_pos is None:
            self.put_obj(Goal(show_goal=self.show_goal), width - 2, height - 2)
        else:
            self.put_obj(
                Goal(show_goal=self.show_goal), self.goal_pos["x"], self.goal_pos["y"]
            )

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def _gen_grid_custom(
        self, width, height, custom_agent_pos=None, custom_agent_dir=None
    ):

        """

        Parameters
        ----------
        width : int
        height : int

        Note that the coordinates used in minigrid is flipped when compared to the tabular gridworld.
        We also need to add the indices by one to consider the wall around the borders.

        Returns
        -------

        """

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.custom_agent_pos = custom_agent_pos
        self.custom_agent_dir = custom_agent_dir

        # Put the goal in the environment
        self.put_obj(
            Goal(show_goal=self.show_goal), self.goal_pos["x"], self.goal_pos["y"]
        )

        # Place the agent
        if self.custom_agent_pos is not None:
            self.agent_pos = self.custom_agent_pos
            self.agent_dir = self.custom_agent_dir
        elif self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def _reward(self):
        """
        Compute the reward to be given upon success, without any penalty
        """

        return 1.0


class EmptyEnv1_5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, goal_pos=dict(x=1, y=1), show_goal=True)


class EmptyEnv1_5x5NoGoalVis(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, goal_pos=dict(x=1, y=1), show_goal=False)


class EmptyEnv2_5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, goal_pos=dict(x=5, y=5), show_goal=True)


class EmptyEnv2_5x5NoGoalVis(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, goal_pos=dict(x=5, y=5), show_goal=False)


class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)


class EmptyEnv6x6(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6)


class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)


class EmptyRandomEnv1(EmptyEnv):
    def __init__(self):
        super().__init__(
            size=12,
            agent_start_pos=None,
            goal_pos=dict(x=1, y=1),
            max_steps=10000,
            show_goal=True,
        )


class EmptyRandomNoGoalVisEnv1(EmptyEnv):
    def __init__(self):
        super().__init__(
            size=12,
            agent_start_pos=None,
            goal_pos=dict(x=1, y=1),
            max_steps=10000,
            show_goal=False,
        )


class EmptyEnv1(EmptyEnv):
    def __init__(self):
        super().__init__(
            size=12,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            goal_pos=dict(x=1, y=1),
            max_steps=10000,
        )


class EmptyRandomEnv2(EmptyEnv):
    def __init__(self):
        super().__init__(
            size=12,
            agent_start_pos=None,
            goal_pos=dict(x=10, y=10),
            max_steps=10000,
            show_goal=True,
        )


class EmptyRandomNoGoalVisEnv2(EmptyEnv):
    def __init__(self):
        super().__init__(
            size=12,
            agent_start_pos=None,
            goal_pos=dict(x=10, y=10),
            max_steps=10000,
            show_goal=False,
        )


class EmptyEnv2(EmptyEnv):
    def __init__(self):
        super().__init__(
            size=12,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            goal_pos=dict(x=10, y=10),
            max_steps=10000,
        )


class EmptyEnv16x16(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)


register(
    id="MiniGrid-Empty-5x5-env1-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv1_5x5",
)

register(
    id="MiniGrid-Empty-Random-5x5-env1-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv1_5x5NoGoalVis",
)

register(
    id="MiniGrid-Empty-5x5-env2-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv2_5x5",
)

register(
    id="MiniGrid-Empty-Random-5x5-env2-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv2_5x5NoGoalVis",
)

register(
    id="MiniGrid-Empty-Random-5x5-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyRandomEnv5x5",
)

register(
    id="MiniGrid-Empty-6x6-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv6x6",
)

register(
    id="MiniGrid-Empty-Random-6x6-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyRandomEnv6x6",
)

register(
    id="MiniGrid-Empty-8x8-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv",
)


register(
    id="MiniGrid-Empty-Random-env1-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyRandomEnv1",
)

register(
    id="MiniGrid-Empty-env1-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv1",
)

register(
    id="MiniGrid-Empty-Random-env2-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyRandomEnv2",
)

register(
    id="MiniGrid-Empty-env2-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv2",
)

register(
    id="MiniGrid-Empty-16x16-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyEnv16x16",
)

register(
    id="MiniGrid-Empty-Random-NoGoalVis-env1-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyRandomNoGoalVisEnv1",
)


register(
    id="MiniGrid-Empty-Random-NoGoalVis-env2-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:EmptyRandomNoGoalVisEnv2",
)
