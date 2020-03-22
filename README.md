# Fractal Exploration Imitation Learning

## **Using the fragile framework as a memory explorer to train a neural network in Atari games**

In this a tutorial we explain how to use the **fragile framework** as an explorer runner useful to memorize high quality memory reply status in order to train a neural network in the OpenAI gym library. It covers how to instantiate all the training process using any Python Jupyter simply running all the cells.

This code has been designed and tested using the Google Colab environment: https://colab.research.google.com/

You should visit and understand before continuing the [getting started tutorial](https://github.com/FragileTech/fragile/blob/master/examples/01_getting_started.ipynb )

## **The main point**

The main point after using here the **fragile framework** is the possibility of training a neural network model in any OpenAI Gym game without the necesity of using a huge random memory reply pack and neither the use of a suplementary target network as usually done in the DQN (Deep Q Learning) reinforcement learning technics.

With the use of the fragile framework we can direclty generate useful and "small" memory reply packs to use directly in the fit process of the model in a supervised learning way.

**Note:**

It's very important to understand that we don't use the reward of every step process. We use a imitation learning method where the model try to imitate what the best fragile framework walker inside the swarm made during its history tree.

## **Results**

This algorithm is able to reach using only a few training runs (and a very small memory reply set) the average score reached by other RL methods like DQN using millions of training steps and a very big memory reply set.

The test was made using the game: **SpaceInvaders**

Human average: ~372

DDQN average: ~479 (128%)

Ours average: ~500

In the game **Atlantis**, our code reach the human average score in more or less 4 training runs: ~25000

## **Note:**

There are even a lot of hyperparameters to play with in order to improve these results ;).
