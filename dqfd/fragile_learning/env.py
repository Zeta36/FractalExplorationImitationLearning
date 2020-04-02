import cv2
from fragile.core import DiscreteEnv
import gym
from gym import spaces
import numpy as np
from PIL import Image
from plangym import AtariEnvironment as PlangymAtari
from plangym.wrappers import FireResetEnv, FrameStack, MaxAndSkipEnv, NoopResetEnv


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(84, 84)):
        super(ProcessFrame84, self).__init__(env)
        self.obs_shape = shape
        self.observation_space = spaces.Box(low=0, high=255, shape=shape)

    def observation(self, obs):
        return self.to_grayscale(obs)

    def to_grayscale(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.obs_shape).convert("L")  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == self.obs_shape
        # saves storage in experience memory
        return processed_observation.astype("uint8")

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


def wrap(env):
    env = NoopResetEnv(env, noop_max=15, override=True)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    # env = MaxAndSkipEnv(env, skip=3)
    # env = FrameStack(env, 4)
    return env


def create_plangym_env(game_name):
    return PlangymAtari(
        name=game_name, min_dt=1, clone_seeds=True, autoreset=True, wrappers=[wrap]
    )


class AtariEnvironment:
    def __new__(cls, game_name):
        env = create_plangym_env(game_name)
        return DiscreteEnv(env=env)
