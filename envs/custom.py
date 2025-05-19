from CRLMSF.environments.gym_minigrid.minigrid import *
from CRLMSF.environments.gym_minigrid.register import register

import itertools as itt

"""
Customize environment for my project
"""


class CustomEnv(MiniGridEnv):
    """
    Environment with wall or lava obstacles, sparse reward.
    """

    def __init__(
        self,
        size=9,
        agent_start_pos=None,
        agent_start_dir=None,
        obstacle_type=Wall,
        env_id=2,
        goal_pos=None,
        max_steps=None,
        show_goal: bool = True,
    ):
        self.obstacle_type = obstacle_type
        self.size = size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_id = env_id
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
            see_through_walls=False,
            seed=None,
            goal_pos=self.goal_pos,
            show_goal=self.show_goal,
        )

    def _gen_grid(self, width, height):

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

        self.midpoint = int(self.size / 2)
        self.obstacles = []

        if self.env_id == 2:
            self.obstacles.append((1, self.midpoint + 1))
            self.obstacles.append((2, self.midpoint + 1))
            for i in range(3, self.midpoint + 2):
                self.obstacles.append((3, i))

        elif self.env_id == 3:
            self.obstacles = [
                (self.size - 2, self.midpoint - 1),
                (self.size - 3, self.midpoint - 1),
            ]

            for i in range(self.midpoint - 1, self.size - 3):
                self.obstacles.append((self.size - 4, i))

        # this part onwards needs to be fixed for coordinates

        elif self.env_id == 4:
            self.obstacles.append((1, self.midpoint + 1))
            self.obstacles.append((2, self.midpoint + 1))

            for i in range(3, self.midpoint + 2):
                self.obstacles.append((3, i))

            for i in range(self.midpoint - 1, self.size - 3):
                self.obstacles.append((self.size - 4, i))

            self.obstacles.append((self.size - 2, self.midpoint - 1))
            self.obstacles.append((self.size - 3, self.midpoint - 1))

        elif self.env_id == 5:
            for i in range(1, self.midpoint - 1):
                self.obstacles.append((i, self.midpoint + 1))

            for i in range(3, self.midpoint + 2):
                self.obstacles.append((self.midpoint - 1, i))

        elif self.env_id == 6:
            for i in range(self.midpoint + 1, self.size):
                self.obstacles.append((i, self.midpoint - 1))

            for i in range(self.midpoint, self.size - 3):
                self.obstacles.append((self.midpoint + 1, i))

        elif self.env_id == 7:
            for i in range(1, self.midpoint - 1):
                self.obstacles.append((i, self.midpoint + 1))

            for i in range(3, self.midpoint + 2):
                self.obstacles.append((self.midpoint - 1, i))

            for i in range(self.midpoint + 1, self.size):
                self.obstacles.append((i, self.midpoint - 1))

            for i in range(self.midpoint, self.size - 3):
                self.obstacles.append((self.midpoint + 1, i))

        elif self.env_id == 8:
            for i in range(1, self.size - 2):
                self.obstacles.append((self.midpoint, i))

        elif self.env_id == 9:
            for i in range(2, self.size - 1):
                self.obstacles.append((self.midpoint, i))

        elif self.env_id == 10:
            for i in range(1, self.midpoint - 2):
                for j in range(4, self.size):
                    self.obstacles.append((i, j))

            for i in range(self.midpoint + 2, self.size):
                for j in range(4, self.size):
                    self.obstacles.append((i, j))

        elif self.env_id == 11:
            for i in range(self.midpoint - 3, self.midpoint + 3):
                for j in range(self.midpoint - 3, self.midpoint + 3):
                    self.obstacles.append((i, j))

        # Add obstacles (Walls) to the environment
        for (x, y) in self.obstacles:
            self.put_obj(self.obstacle_type(), x, y)

        # Put the goal in the environment
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

        self.custom_agent_pos = custom_agent_pos
        self.custom_agent_dir = custom_agent_dir

        self.midpoint = int(self.size / 2)
        self.obstacles = []

        if self.env_id == 2:
            self.obstacles.append((1, self.midpoint + 1))
            self.obstacles.append((2, self.midpoint + 1))
            for i in range(3, self.midpoint + 2):
                self.obstacles.append((3, i))

        elif self.env_id == 3:
            self.obstacles.append((self.size - 2, self.midpoint - 1))
            self.obstacles.append((self.size - 3, self.midpoint - 1))
            for i in range(self.midpoint - 1, self.size - 3):
                self.obstacles.append((self.size - 4, i))

        # this part onwards needs to be fixed for coordinates
        elif self.env_id == 4:
            self.obstacles.append((1, self.midpoint + 1))
            self.obstacles.append((2, self.midpoint + 1))

            for i in range(3, self.midpoint + 2):
                self.obstacles.append((3, i))

            for i in range(self.midpoint - 1, self.size - 3):
                self.obstacles.append((self.size - 4, i))

            self.obstacles.append((self.size - 2, self.midpoint - 1))
            self.obstacles.append((self.size - 3, self.midpoint - 1))

        elif self.env_id == 5:
            for i in range(1, self.midpoint - 1):
                self.obstacles.append((i, self.midpoint + 1))

            for i in range(3, self.midpoint + 2):
                self.obstacles.append((self.midpoint - 1, i))

        elif self.env_id == 6:
            for i in range(self.midpoint + 1, self.size):
                self.obstacles.append((i, self.midpoint - 1))

            for i in range(self.midpoint, self.size - 3):
                self.obstacles.append((self.midpoint + 1, i))

        elif self.env_id == 7:
            for i in range(1, self.midpoint - 1):
                self.obstacles.append((i, self.midpoint + 1))

            for i in range(3, self.midpoint + 2):
                self.obstacles.append((self.midpoint - 1, i))

            for i in range(self.midpoint + 1, self.size):
                self.obstacles.append((i, self.midpoint - 1))

            for i in range(self.midpoint, self.size - 3):
                self.obstacles.append((self.midpoint + 1, i))

        elif self.env_id == 8:
            for i in range(1, self.size - 2):
                self.obstacles.append((self.midpoint, i))

        elif self.env_id == 9:
            for i in range(2, self.size - 1):
                self.obstacles.append((self.midpoint, i))

        elif self.env_id == 10:
            for i in range(1, self.midpoint - 2):

                for j in range(4, self.size):
                    self.obstacles.append((i, j))

            for i in range(self.midpoint + 2, self.size):

                for j in range(4, self.size):
                    self.obstacles.append((i, j))

        elif self.env_id == 11:
            for i in range(self.midpoint - 3, self.midpoint + 3):
                for j in range(self.midpoint - 3, self.midpoint + 3):
                    self.obstacles.append((i, j))

        # Add obstacles (Walls) to the environment
        for (x, y) in self.obstacles:
            self.put_obj(self.obstacle_type(), x, y)

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
        Compute the reward to be given upon success
        """

        return 1.0

    def step(self, action):
        self.step_count += 1

        reward = 0.0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()

        return (
            obs,
            reward,
            terminated,
            truncated,
            {"agent_pos": self.agent_pos, "agent_dir": self.agent_dir},
        )


class GridworldEnv2(CustomEnv):
    def __init__(self):
        super().__init__(size=12, env_id=2, goal_pos=dict(x=1, y=6), max_steps=10000)


class GridworldEnv3(CustomEnv):
    def __init__(self):
        super().__init__(size=12, env_id=3, goal_pos=dict(x=10, y=6), max_steps=10000)


class GridworldEnv4GoalLeft(CustomEnv):
    def __init__(self):
        super().__init__(size=12, env_id=4, goal_pos=dict(x=1, y=6), max_steps=10000)


class GridworldEnv4GoalRight(CustomEnv):
    def __init__(self):
        super().__init__(size=12, env_id=4, goal_pos=dict(x=10, y=6), max_steps=10000)


class GridworldEnv5(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=5, goal_pos=dict(x=1, y=6), max_steps=10000, show_goal=True
        )


class GridworldEnv5NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=5, goal_pos=dict(x=1, y=6), max_steps=10000, show_goal=False
        )


class GridworldEnv6GoalLeft(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=6, goal_pos=dict(x=1, y=6), max_steps=10000, show_goal=True
        )


class GridworldEnv6NoGoalVisGoalLeft(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=6, goal_pos=dict(x=1, y=6), max_steps=10000, show_goal=False
        )


class GridworldEnv6(CustomEnv):
    def __init__(self):
        super().__init__(size=12, env_id=6, goal_pos=dict(x=10, y=6), max_steps=10000)


class GridworldEnv7GoalLeft(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=7, goal_pos=dict(x=1, y=6), max_steps=10000, show_goal=True
        )


class GridworldEnv7GoalRight(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=7, goal_pos=dict(x=10, y=6), max_steps=10000, show_goal=True
        )


class GridworldEnv7NoGoalVisGoalLeft(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=7, goal_pos=dict(x=1, y=6), max_steps=10000, show_goal=False
        )


class GridworldEnv7NoGoalVisGoalRight(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12,
            env_id=7,
            goal_pos=dict(x=10, y=6),
            max_steps=10000,
            show_goal=False,
        )


class GridworldEnv8(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=8, goal_pos=dict(x=1, y=1), max_steps=10000, show_goal=True
        )


class GridworldEnv8NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=8, goal_pos=dict(x=1, y=1), max_steps=10000, show_goal=False
        )


class GridworldEnv8_5x5(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7, env_id=8, goal_pos=dict(x=1, y=1), max_steps=10000, show_goal=True
        )


class GridworldEnv8_FixStart_5x5(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7,
            env_id=8,
            goal_pos=dict(x=1, y=1),
            max_steps=10000,
            show_goal=True,
            agent_start_pos=(5, 1),
            agent_start_dir=1,
        )


class GridworldEnv8_5x5NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7, env_id=8, goal_pos=dict(x=1, y=1), max_steps=10000, show_goal=False
        )


class GridworldEnv8__FixStart_5x5NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7,
            env_id=8,
            goal_pos=dict(x=1, y=1),
            max_steps=10000,
            show_goal=False,
            agent_start_pos=(5, 1),
            agent_start_dir=1,
        )


class GridworldEnv9(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12,
            env_id=9,
            goal_pos=dict(x=10, y=10),
            max_steps=10000,
            show_goal=True,
        )


class GridworldEnv9NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12,
            env_id=9,
            goal_pos=dict(x=10, y=10),
            max_steps=10000,
            show_goal=False,
        )


class GridworldEnv9_5x5(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7,
            env_id=9,
            goal_pos=dict(x=5, y=5),
            max_steps=10000,
            show_goal=True,
        )

class GridworldEnv9_FixStart_5x5(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7,
            env_id=9,
            goal_pos=dict(x=5, y=5),
            max_steps=10000,
            show_goal=True,
            agent_start_pos=(5, 1),
            agent_start_dir=1,
        )


class GridworldEnv9_5x5NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7,
            env_id=9,
            goal_pos=dict(x=5, y=5),
            max_steps=10000,
            show_goal=False,
        )

class GridworldEnv9_FixStart_5x5NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=7,
            env_id=9,
            goal_pos=dict(x=5, y=5),
            max_steps=10000,
            show_goal=False,
            agent_start_pos=(5, 1),
            agent_start_dir=1,
        )


class GridworldEnv10(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=10, goal_pos=dict(x=1, y=1), max_steps=10000, show_goal=True
        )


class GridworldEnv10NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12,
            env_id=10,
            goal_pos=dict(x=1, y=1),
            max_steps=10000,
            show_goal=False,
        )


class GridworldEnv11(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12, env_id=11, goal_pos=dict(x=1, y=1), max_steps=10000, show_goal=True
        )


class GridworldEnv11NoGoalVis(CustomEnv):
    def __init__(self):
        super().__init__(
            size=12,
            env_id=11,
            goal_pos=dict(x=1, y=1),
            max_steps=10000,
            show_goal=False,
        )


class GridworldEnv12(CustomEnv):
    """A 3x3 gridworld with a goal in the top left corner."""

    def __init__(self):
        super().__init__(
            size=5, env_id=8, goal_pos=dict(x=1, y=1), max_steps=10000, show_goal=True
        )


register(
    id="MiniGrid-GridworldEnv2-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv2",
)

register(
    id="MiniGrid-GridworldEnv3-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv3",
)

register(
    id="MiniGrid-GridworldEnv4-GoalLeft-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv4GoalLeft",
)

register(
    id="MiniGrid-GridworldEnv4-GoalRight-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv4GoalRight",
)

register(
    id="MiniGrid-GridworldEnv5-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv5",
)

register(
    id="MiniGrid-GridworldEnv5-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv5NoGoalVis",
)

register(
    id="MiniGrid-GridworldEnv6-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv6",
)

register(
    id="MiniGrid-GridworldEnv6-GoalLeft-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv6GoalLeft",
)

register(
    id="MiniGrid-GridworldEnv6-NoGoalVis-GoalLeft-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv6NoGoalVisGoalLeft",
)

register(
    id="MiniGrid-GridworldEnv7-GoalLeft-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv7GoalLeft",
)

register(
    id="MiniGrid-GridworldEnv7-GoalRight-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv7GoalRight",
)

register(
    id="MiniGrid-GridworldEnv7-NoGoalVis-GoalLeft-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv7NoGoalVisGoalLeft",
)

register(
    id="MiniGrid-GridworldEnv7-NoGoalVis-GoalRight-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv7NoGoalVisGoalRight",
)

register(
    id="MiniGrid-GridworldEnv8-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv8",
)

register(
    id="MiniGrid-GridworldEnv8-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv8NoGoalVis",
)

register(
    id="MiniGrid-GridworldEnv8-5x5-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv8_5x5",
)

register(
    id="MiniGrid-GridworldEnv8-FixStartPos-5x5-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv8_FixStart_5x5",
)

register(
    id="MiniGrid-GridworldEnv8-5x5-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv8_5x5NoGoalVis",
)

register(
    id="MiniGrid-GridworldEnv8-FixStartPos-5x5-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv8_FixStart_5x5NoGoalVis",
)

register(
    id="MiniGrid-GridworldEnv9-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv9",
)


register(
    id="MiniGrid-GridworldEnv9-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv9NoGoalVis",
)

register(
    id="MiniGrid-GridworldEnv9-5x5-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv9_5x5",
)

register(
    id="MiniGrid-GridworldEnv9-FixStartPos-5x5-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv9_FixStart_5x5",
)

register(
    id="MiniGrid-GridworldEnv9-5x5-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv9_5x5NoGoalVis",
)

register(
    id="MiniGrid-GridworldEnv9-FixStartPos-5x5-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv9_FixStart_5x5NoGoalVis",
)

register(
    id="MiniGrid-TMaze-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv10",
)

register(
    id="MiniGrid-TMaze-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv10NoGoalVis",
)

register(
    id="MiniGrid-SquareArena-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv11",
)

register(
    id="MiniGrid-SquareArena-NoGoalVis-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv11NoGoalVis",
)

register(
    id="MiniGrid-3x3-v0",
    entry_point="CRLMSF.environments.gym_minigrid.envs:GridworldEnv12",
)
