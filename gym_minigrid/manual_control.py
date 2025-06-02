#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import random
import gym_minigrid

from window import Window
from wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
# from environments import gym_minigrid
# from environments.gym_minigrid.wrappers import *
# from environments.gym_minigrid.window import Window

# from environments.gym_minigrid.wrappers import *
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
import sys
import jax
import optax
import chex

from scipy import stats

from absl import app
from absl.flags import argparse_flags


class manual_control:
    def __init__(
        self,
        args,
        window: Window = None,
    ):
        self.args = args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        game = self.args.env
        # self.env = ImgObsWrapper(RGBImgPartialObsWrapper(gym.make(game, disable_env_checker=True), tile_size=32))

        if self.args.agent_view:
            self.env = RGBImgPartialObsWrapper(
                gym.make(game, disable_env_checker=True),
                tile_size=self.args.tile_size,
            )
        else:
            self.env = ImgObsWrapper(gym.make(game, disable_env_checker=True), tile_size=self.args.tile_size)

        if window is None:
            window = Window("minigrid - " + str(self.env.__class__))
        self.window = window
        self.window.reg_key_handler(self.key_handler)
        self.num_actions = self.env.action_space.n
        self.sf_u1 = None
        self.sf_u2 = None
        self.sf_u3 = None
        self.sf_u1_compressed = []
        self.sf_u2_compressed = []
        self.sf_u3_compressed = []

        self.agent_pos_x = []
        self.agent_pos_y = []
        self.agent_head_dir = []
        self.hd_markers = []

    def start(self):
        print("I am starting!")
        self.reset()
        self.window.show(block=True)

    def reset(self):
        self.env.reset()
        if hasattr(self.env, "mission"):
            print("Mission: %s" % self.env.mission)
            self.window.set_caption(self.env.mission)

        self.redraw()

    def redraw(self):
        if not self.args.agent_view:
            img = self.env.render(mode="rgb_array", tile_size=self.args.tile_size, highlight=False)

        else:
            img = self.env.render(
                mode="rgb_array",
                tile_size=self.args.tile_size,
                highlight=True,
            )

        self.window.show_img(img)

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset()
        else:
            self.redraw()

        return obs, reward, terminated, truncated, info

    # def make_actions_vector(self):
    #     action_inputs = None
    #
    #     for action in range(self.num_actions):
    #         one_hot = np.zeros(self.num_actions)
    #         one_hot[action] = 1
    #         if action_inputs is None:
    #             action_inputs = one_hot
    #         else:
    #             if action_inputs.ndim == 1:
    #                 action_inputs = np.stack((action_inputs, one_hot))
    #             else:
    #                 action_inputs = np.concatenate(
    #                     (action_inputs, np.expand_dims(one_hot, axis=0))
    #                 )
    #
    #     return action_inputs

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.window.close()
            return
        if key == "backspace":
            self.reset()
            return

        if key == "enter":
            print("return button pressed!!!")
            return

        if key == " ":
            print("space pressed!!")
            self.window.close()
            return

        key_to_action = {
            "left": self.env.actions.left,
            "right": self.env.actions.right,
            "up": self.env.actions.forward,
        }

        action = key_to_action[key]
        self.step(action)


def parse_flags(argv):
    parser = argparse_flags.ArgumentParser(
        description="absl.flags and argparse integration."
    )

    parser.add_argument(
        "--env", help="gym environment to load", default="MiniGrid-Gridworld-VertWallTop4-20x20-v0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=97,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )

    parser.add_argument(
        "--grid_size", type=int, help="size of the grid on each side", default=10
    )

    parser.add_argument(
        "--environment_height",
        type=int,
        help="size at which to render tiles",
        default=96,
    )

    parser.add_argument(
        "--environment_width",
        type=int,
        help="size at which to render tiles",
        default=96,
    )

    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    parser.add_argument(
        "--num_stacked_frames",
        type=int,
        help="number of channels for each observation (3 for RGB, 4 Atari stacked frames)",
        default=3,
    )

    parser.add_argument(
        "--eval_exploration_epsilon",
        type=float,
        help="epsilon-greedy value for evaluation",
        default=0.05,
    )

    return parser.parse_args(argv[1:])


def main(args):
    mc = manual_control(args)
    mc.start()


if __name__ == "__main__":
    # args = parse_flags("")
    # MC = manual_control_sf(args)
    # MC.start()
    app.run(main, flags_parser=parse_flags)
