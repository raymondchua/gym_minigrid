from gym_minigrid.minigrid import *
from gym_minigrid.register import register

import itertools as itt

"""
Customize environment for my project
"""


class VertWallBottom3Env(MiniGridEnv):
    """
    Environment with wall sparse reward. The goal is a green square at bottom right corner with a reward of 1.0.
    """

    def __init__(
        self,
        size=9,
        agent_start_pos=None,
        agent_start_dir=None,
        obstacle_type=Wall,
        goal_pos=None,
        max_steps=None,
        show_goal: bool = True,
    ):
        self.obstacle_type = obstacle_type
        self.size = size
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

        for i in range(2, self.size - 1):
            self.obstacles.append((self.midpoint, i))

        # Add obstacles (Walls) to the environment
        for (x, y) in self.obstacles:
            self.put_obj(self.obstacle_type(), x, y)

        # Add lava on the left side of the environment
        self.put_obj(Lava(), 2, 4)
        self.put_obj(Lava(), 2, 5)
        self.put_obj(Lava(), 3, 5)
        self.put_obj(Lava(), 3, 8)
        self.put_obj(Lava(), 4, 8)

        # Add lava on the right side of the environment
        self.put_obj(Lava(), 9, 2)
        self.put_obj(Lava(), 8, 6)
        self.put_obj(Lava(), 8, 7)
        self.put_obj(Lava(), 8, 8)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green box"

        if self.goal_pos is not None:
            self.put_obj(
                Goal(show_goal=self.show_goal), self.goal_pos["x"], self.goal_pos["y"]
            )

        else:
            # place goal randomly in the environment
            goal_pos_x = np.random.randint(1, self.size - 1)
            goal_pos_y = np.random.randint(1, self.size - 1)

            # check if we can place goal in the randomly selected position
            while (goal_pos_x, goal_pos_y) in self.obstacles:
                goal_pos_x = np.random.randint(1, self.size - 1)
                goal_pos_y = np.random.randint(1, self.size - 1)

            self.put_obj(Goal(show_goal=self.show_goal), goal_pos_x, goal_pos_y)

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

        self.midpoint = int(self.size / 2)
        self.obstacles = []

        for i in range(2, self.size - 1):
            self.obstacles.append((self.midpoint, i))

        # Add obstacles (Walls) to the environment
        for (x, y) in self.obstacles:
            self.put_obj(self.obstacle_type(), x, y)

        # Add lava on the left side of the environment
        self.put_obj(Lava(), 2, 4)
        self.put_obj(Lava(), 2, 5)
        self.put_obj(Lava(), 3, 5)
        self.put_obj(Lava(), 3, 8)
        self.put_obj(Lava(), 4, 8)

        # Add lava on the right side of the environment
        self.put_obj(Lava(), 9, 2)
        self.put_obj(Lava(), 8, 6)
        self.put_obj(Lava(), 8, 7)
        self.put_obj(Lava(), 8, 8)

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

        if self.goal_pos is not None:
            self.put_obj(
                Goal(show_goal=self.show_goal), self.goal_pos["x"], self.goal_pos["y"]
            )

        else:
            # place goal randomly in the environment
            goal_pos_x = np.random.randint(1, self.size - 1)
            goal_pos_y = np.random.randint(1, self.size - 1)

            # check if we can place goal in the randomly selected position
            while (goal_pos_x, goal_pos_y) in self.obstacles:
                goal_pos_x = np.random.randint(1, self.size - 1)
                goal_pos_y = np.random.randint(1, self.size - 1)

            self.put_obj(Goal(show_goal=self.show_goal), goal_pos_x, goal_pos_y)

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
                reward = 1.0

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


class GridworldVertWallBottom3(VertWallBottom3Env):
    def __init__(self):
        super().__init__(
            size=12,
            goal_pos=dict(x=10, y=10),
            show_goal=True,
        )


class GridworldVertWallBottom3NoGoalVis(VertWallBottom3Env):
    def __init__(self):
        super().__init__(
            size=12,
            goal_pos=dict(x=10, y=10),
            show_goal=False,
        )


class GridworldVertWallBottom3SameStartPos(VertWallBottom3Env):
    def __init__(self):
        super().__init__(
            size=12,
            goal_pos=dict(x=10, y=10),
            show_goal=True,
            agent_start_pos=(5, 6),  # top right corner
            agent_start_dir=1,  # facing down
        )


class GridworldVertWallBottom3NoGoalVisSameStartPos(VertWallBottom3Env):
    def __init__(self):
        super().__init__(
            size=12,
            goal_pos=dict(x=10, y=10),
            show_goal=False,
            agent_start_pos=(5, 6),  # top right corner
            agent_start_dir=1,  # facing down
        )


class GridworldVertWallBottom3RandomGoal(VertWallBottom3Env):
    def __init__(self):
        super().__init__(
            size=12,
            goal_pos=None,
            show_goal=True,
        )


class GridworldVertWallBottom3RandomGoalNoGoalVis(VertWallBottom3Env):
    def __init__(self):
        super().__init__(
            size=12,
            goal_pos=None,
            show_goal=False,
        )


register(
    id="MiniGrid-Gridworld-VertWallBottom3-v0",
    entry_point="gym_minigrid.envs:GridworldVertWallBottom3",
)

register(
    id="MiniGrid-Gridworld-VertWallBottom3-NoGoalVis-v0",
    entry_point="gym_minigrid.envs:GridworldVertWallBottom3NoGoalVis",
)


register(
    id="MiniGrid-Gridworld-VertWallBottom3-SameStartPos-v0",
    entry_point="gym_minigrid.envs:GridworldVertWallBottom3SameStartPos",
)

register(
    id="MiniGrid-Gridworld-VertWallBottom3-NoGoalVis-SameStartPos-v0",
    entry_point="gym_minigrid.envs:GridworldVertWallBottom3NoGoalVisSameStartPos",
)

register(
    id="MiniGrid-Gridworld-VertWallBottom3-RandomGoal-v0",
    entry_point="gym_minigrid.envs:GridworldVertWallBottom3RandomGoal",
)

register(
    id="MiniGrid-Gridworld-VertWallBottom3-RandomGoal-NoGoalVis-v0",
    entry_point="gym_minigrid.envs:GridworldVertWallBottom3RandomGoalNoGoalVis",
)
