import argparse
import sys
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.optimizers import Adam

tf.disable_v2_behavior()

from dqfd.kerasrl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from dqfd.kerasrl.policy import EpsGreedyQPolicy

from dqfd.fragile_learning.env import create_plangym_env
from dqfd.fragile_learning.memory import PartitionedMemory
from dqfd.fragile_learning.runner import FragileRunner

from dqfd.agent import AtariDQfDProcessor, AtariDQfDTestProcessor, DQfDAgent
from dqfd.model import DQFDNeuralNet


def main():
    # We downsize the atari frame to 84 x 84 and feed the model 4 frames at a time for
    # a sense of direction and speed.
    INPUT_SHAPE = (84, 84)
    WINDOW_LENGTH = 4
    # Runner parameters
    EXPLORE_MEMORY_STEPS = 5
    fractal_memory_size = 10000
    n_walkers = 32
    n_workers = 8
    max_epochs_per_game = 2000
    score_limit_per_game = 1500
    # Training parameters
    n_training_steps = 1000
    pretraining_steps = 75000
    target_model_update = 10000
    n_max_episode_steps = 10000000
    rl_training_memory_max_size = 100000
    # testing
    n_episodes_test = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--env-name", type=str, default="SpaceInvaders-v0")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    # Get the environment and extract the number of actions.
    env = create_plangym_env(args.env_name)
    n_actions = env.action_space.n

    if args.mode == "test":
        processor = AtariDQfDTestProcessor()
    else:
        processor = AtariDQfDProcessor()

    explorer = FragileRunner(
        args.env_name,
        memory_size=fractal_memory_size,
        n_walkers=n_walkers,
        n_workers=n_workers,
        max_epochs=max_epochs_per_game,
        score_limit=score_limit_per_game,
    )
    explorer.run()
    processed_memory = processor.process_demo_data(explorer.memory)

    memory = PartitionedMemory(
        limit=rl_training_memory_max_size,
        swarm_memory=processed_memory,
        alpha=0.4,
        start_beta=0.6,
        end_beta=0.6,
        window_length=WINDOW_LENGTH,
    )

    policy = EpsGreedyQPolicy(0.01)

    model = DQFDNeuralNet(
        window_length=WINDOW_LENGTH, n_actions=explorer.n_actions, input_shape=INPUT_SHAPE
    )
    dqfd = DQfDAgent(
        model=model,
        n_actions=n_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        target_model_update=target_model_update,
        pretraining_steps=pretraining_steps,
        enable_double_dqn=True,
        enable_dueling_network=True,
        gamma=0.99,
        train_interval=4,
        delta_clip=1.0,
        n_step=10,
    )

    lr = 0.00025 / 4
    dqfd.compile(Adam(lr=lr), metrics=["mae"])

    if args.mode == "train":
        weights_filename = "dqfd_{}_weights.h5f".format(args.env_name)
        checkpoint_weights_filename = "dqfd_" + args.env_name + "_weights_{step}.h5f"
        # uses TrainEpisodeLogger csv (optional)
        log_filename = "dqfd_" + args.env_name + "_REWARD_DATA.txt"
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000)]
        callbacks += [TrainEpisodeLogger(log_filename)]
        dqfd.fit(
            env,
            callbacks=callbacks,
            nb_steps=10000000,
            verbose=0,
            n_max_episode_steps=n_max_episode_steps,
        )
        dqfd.save_weights(weights_filename, overwrite=True)

    elif args.mode == "test":
        weights_filename = "dqfd_{}_weights.h5f".format(args.env_name)
        if args.weights:
            weights_filename = args.weights
        dqfd.load_weights(weights_filename)
        dqfd.test(env, n_episodes=n_episodes_test, visualize=False)


if __name__ == "__main__":
    sys.exit(main())
