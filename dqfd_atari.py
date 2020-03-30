from __future__ import division
from fragile.core.utils import remove_notebook_margin
import holoviews
from fragile.dataviz.swarm_viz import AtariViz
from fragile.core.swarm import Swarm
from fragile.core.tree import HistoryTree
from fragile.core.models import DiscreteUniform
from fragile.core.dt_sampler import GaussianDt
from fragile.atari.env import AtariEnv
from plangym import AtariEnvironment, ParallelEnvironment
from rl.util import load_demo_data_from_file, record_demo_data
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import PartitionedMemory
from rl.policy import EpsGreedyQPolicy
from rl.agents.dqn import DQfDAgent
import argparse
from PIL import Image
import numpy as np
import gym
import time

from keras.models import Model
from keras.layers import Flatten, Conv2D, Input, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
import sys
sys.path.append('rl')


# We downsize the atari frame to 84 x 84 and feed the model 4 frames at a time for
# a sense of direction and speed.
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
EXPLORE_MEMORY_STEPS = 5
MEMORY_SIZE = 75000


class FragileRunner:
    def __init__(self, game_name):

        self.env = ParallelEnvironment(
            env_class=AtariEnvironment,
            name=game_name,
            clone_seeds=True,
            autoreset=True,
            blocking=False,
        )

        self.game_name = game_name
        self.env_callable = lambda: AtariEnv(env=self.env)
        self.dt = GaussianDt(min_dt=3, max_dt=1000, loc_dt=4, scale_dt=2)
        self.model_callable = lambda env: DiscreteUniform(
            env=self.env, critic=self.dt)
        self.prune_tree = True
        # A bigger number will increase the quality of the trajectories sampled.
        self.n_walkers = 16
        self.max_iters = 1000  # Increase to sample longer games.
        self.reward_scale = 2  # Rewards are more important than diversity.
        self.distance_scale = 1
        self.minimize = False  # We want to get the maximum score possible.
        self.memory = []

    def run(self):

        swarm = Swarm(
            model=self.model_callable,
            env=self.env_callable,
            tree=HistoryTree,
            n_walkers=self.n_walkers,
            max_iters=self.max_iters,
            prune_tree=self.prune_tree,
            reward_scale=self.reward_scale,
            distance_scale=self.distance_scale,
            minimize=self.minimize,
        )

        holoviews.extension("bokeh")
        swarm_viz = AtariViz(swarm)
        swarm_viz.plot()

        print("Creating fractal replay memory...")

        while len(self.memory) < MEMORY_SIZE:
            try:
                _ = swarm_viz.run(print_every=1000)

                leafs = swarm.tree.get_leaf_nodes()

                env_name = self.game_name
                env = gym.make(env_name)

                for i in range(len(leafs)):
                    path = swarm.tree.get_branch(leafs[i], from_hash=False)
                    current_state = env.reset()
                    for a in path[1]:

                        next_state, reward, terminal, _ = env.step(a)

                        self.memory.append(
                            [current_state, a, reward, terminal])

                        if len(self.memory) > MEMORY_SIZE:
                            self.memory.pop(0)
                        time.sleep(0.00005)
                        current_state = next_state

                print("Fractal replay memory size: ", len(self.memory))

            except:
                pass

        return self.memory


# Standard Atari processing
class AtariDQfDProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert(
            'L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.sign(reward) * np.log(1 + abs(reward))

    def process_demo_data(self, demo_data):
        # Important addition from dqn example.
        for step in demo_data:
            step[0] = self.process_observation(step[0])
            step[2] = self.process_reward(step[2])
        return demo_data


class AtariDQfDTestProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert(
            'L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        # saves storage in experience memory
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return reward

    def process_demo_data(self, demo_data):
        # Important addition from dqn example.
        for step in demo_data:
            step[0] = self.process_observation(step[0])
            step[2] = self.process_reward(step[2])
        return demo_data


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='SpaceInvaders-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(231)
env.seed(123)
nb_actions = env.action_space.n
print("NUMBER OF ACTIONS: " + str(nb_actions))

# Standard DQN model architecture + l2 regularization to prevent overfitting on small demo sets.
input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])
frame = Input(shape=(input_shape))
cv1 = Conv2D(32, kernel_size=(8, 8), strides=4, activation='relu',
             kernel_regularizer=l2(1e-4), data_format='channels_first')(frame)
cv2 = Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu',
             kernel_regularizer=l2(1e-4), data_format='channels_first')(cv1)
cv3 = Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu',
             kernel_regularizer=l2(1e-4), data_format='channels_first')(cv2)
dense = Flatten()(cv3)
dense = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(dense)
buttons = Dense(nb_actions, activation='linear',
                kernel_regularizer=l2(1e-4))(dense)
model = Model(inputs=frame, outputs=buttons)
model.summary()

if args.mode == 'test':
    processor = AtariDQfDTestProcessor()
else:
    processor = AtariDQfDProcessor()

explorer = FragileRunner(args.env_name)
expert_demo_data = explorer.run()
expert_demo_data = processor.process_demo_data(expert_demo_data)

memory = PartitionedMemory(limit=1000000, pre_load_data=expert_demo_data,
                           alpha=.4, start_beta=.6, end_beta=.6, window_length=WINDOW_LENGTH)

policy = EpsGreedyQPolicy(.01)

dqfd = DQfDAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                 processor=processor, enable_double_dqn=True, enable_dueling_network=True, gamma=.99, target_model_update=10000,
                 train_interval=4, delta_clip=1., pretraining_steps=75000, n_step=10)

lr = .00025/4
dqfd.compile(Adam(lr=lr), metrics=['mae'])

if args.mode == 'train':
    weights_filename = 'dqfd_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqfd_' + \
        args.env_name + '_weights_{step}.h5f'
    # uses TrainEpisodeLogger csv (optional)
    log_filename = 'dqfd_' + args.env_name + '_REWARD_DATA.txt'
    callbacks = [ModelIntervalCheckpoint(
        checkpoint_weights_filename, interval=500000)]
    callbacks += [TrainEpisodeLogger(log_filename)]
    dqfd.fit(env, callbacks=callbacks, nb_steps=10000000,
             verbose=0, nb_max_episode_steps=200000)
    dqfd.save_weights(weights_filename, overwrite=True)

elif args.mode == 'test':
    weights_filename = 'dqfd_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqfd.load_weights(weights_filename)
    dqfd.test(env, nb_episodes=10, visualize=False)
