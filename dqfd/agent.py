import warnings

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda, Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.generic_utils import Progbar

from dqfd.kerasrl.agents.dqn import mean_q
from dqfd.kerasrl.core import Agent, Processor
from dqfd.kerasrl.layers import NoisyNetDense
from dqfd.kerasrl.memory import PartitionedMemory
from dqfd.kerasrl.policy import EpsGreedyQPolicy, GreedyQPolicy
from dqfd.kerasrl.util import (
    get_object_config,
    AdditionalUpdatesOptimizer,
    get_soft_target_model_updates,
    huber_loss,
    clone_model,
)


class AtariDQfDProcessor(Processor):
    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        processed_batch = batch.astype("float32") / 255.0
        return np.transpose(processed_batch, axes=(0, 2, 3, 1))

    def process_reward(self, reward):
        return np.sign(reward) * np.log(1 + np.abs(reward))

    def process_demo_data(self, swarm_memory):
        # Important addition from dqn example.
        swarm_memory.rewards = self.process_reward(swarm_memory.rewards)
        return swarm_memory


class AtariDQfDTestProcessor(Processor):
    def process_observation(self, observation):
        return observation

    def process_state_batch(self, batch):
        processed_batch = batch.astype("float32") / 255.0
        return np.transpose(processed_batch, axes=(0, 2, 3, 1))

    def process_reward(self, reward):
        return reward

    def process_demo_data(self, swarm_memory):
        # Important addition from dqn example.
        swarm_memory.rewards = self.process_reward(swarm_memory.rewards)
        return swarm_memory


class AbstractDQNAgent(Agent):
    """Write me
    """

    def __init__(
        self,
        n_actions,
        memory,
        gamma=0.99,
        batch_size=32,
        n_steps_warmup=1000,
        train_interval=1,
        memory_interval=1,
        target_model_update=10000,
        delta_range=None,
        delta_clip=np.inf,
        custom_model_objects={},
        **kwargs
    ):
        super(AbstractDQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError("`target_model_update` must be >= 0.")
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn(
                "`delta_range` is deprecated. Please use `delta_clip` instead, which "
                "takes a single scalar. For now we're falling back "
                "to `delta_range[1] = {}`".format(delta_range[1])
            )
            delta_clip = delta_range[1]

        # Parameters.
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_steps_warmup = n_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = custom_model_objects
        # Related objects.
        self.memory = memory
        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (q_values.shape[0], self.n_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.n_actions,)
        return q_values

    def get_config(self):
        return {
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "n_steps_warmup": self.n_steps_warmup,
            "train_interval": self.train_interval,
            "memory_interval": self.memory_interval,
            "target_model_update": self.target_model_update,
            "delta_clip": self.delta_clip,
            "memory": get_object_config(self.memory),
        }

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def _calc_double_q_values(self, state1_batch):
        # According to the paper "Deep Reinforcement Learning with Double Q-learning"
        # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
        # while the target network is used to estimate the Q value.
        q_values = self.model.predict_on_batch(state1_batch)
        assert q_values.shape == (self.batch_size, self.n_actions)
        actions = np.argmax(q_values, axis=1)
        assert actions.shape == (self.batch_size,)

        # Now, estimate Q values using the target network but select the values with the
        # highest Q value wrt to the online model (as computed above).
        target_q_values = self.target_model.predict_on_batch(state1_batch)
        assert target_q_values.shape == (self.batch_size, self.n_actions)
        q_batch = target_q_values[range(self.batch_size), actions]

        return q_batch

    def _build_dueling_arch(self, model, dueling_type):
        # bulid the two-stream architecture
        layer = model.layers[-2]
        n_action = model.output.shape[-1]
        y = Dense(n_action + 1, activation="linear")(layer.output)
        # preserve use of noisy nets.
        if isinstance(layer, NoisyNetDense):
            y = NoisyNetDense(n_action + 1, activation="linear")(layer.output)
        # options for dual-stream merger
        if dueling_type == "avg":
            outputlayer = Lambda(
                lambda a: K.expand_dims(a[:, 0], -1)
                + a[:, 1:]
                - K.mean(a[:, 1:], axis=1, keepdims=True),
                output_shape=(n_action,),
            )(y)
        elif dueling_type == "max":
            outputlayer = Lambda(
                lambda a: K.expand_dims(a[:, 0], -1)
                + a[:, 1:]
                - K.max(a[:, 1:], axis=1, keepdims=True),
                output_shape=(n_action,),
            )(y)
        elif dueling_type == "naive":
            outputlayer = Lambda(
                lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(n_action,),
            )(y)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        return Model(inputs=model.input, outputs=outputlayer)


class DQfDAgent(AbstractDQNAgent):
    def __init__(
        self,
        model,
        policy=None,
        test_policy=None,
        enable_double_dqn=True,
        enable_dueling_network=True,
        dueling_type="avg",
        n_step=10,
        pretraining_steps=750000,
        large_margin=0.8,
        lam_2=1.0,
        *args,
        **kwargs
    ):
        """
        Deep Q-Learning from Demonstrations. Uses demonstrations from an expert controller to kickstart training and improve
        sample efficiency. [paper](https://arxiv.org/abs/1704.03732).

        model__: A Keras model.
        policy__: A Keras-kerasrl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-kerasrl policy.
        enable_double_dqn__: A boolean which enables target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enables dueling architecture proposed by Wang et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
        nb_actions__: The total number of actions the agent can take. Dependent on the environment.
        processor__: A Keras-kerasrl processor. An intermediary between the environment and the agent. Resizes the input, clips rewards etc. Similar to gym env wrappers.
        nb_steps_warmup__: An integer number of random steps to take before learning begins. This puts experience into the memory.
        gamma__: The discount factor of future rewards in the Q function.
        target_model_update__: How often to update the target model. Longer intervals stabilize training.
        train_interval__: The integer number of steps between each learning process.
        delta_clip__: A component of the huber loss.
        n_step__: exponent for multi-step learning. Larger values extend the future reward approximations further into the future.
        pretraining_steps__: Length of 'pretraining' in which the agent learns exclusively from the expert demonstration data.
        large_margin__: Constant value that pushes loss of incorrect action choices a margin higher than the others.
        lam_2__: Imitation loss coefficient.
        """

        super(DQfDAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        if hasattr(model.output, "__len__") and len(model.output.shape) > 2:
            raise ValueError(
                'Model "{}" has more than one output. DQfD expects a model that'
                " has a single output.".format(model)
            )
        if model.output.shape[1] != self.n_actions:
            raise ValueError(
                'Model output "{}" has invalid shape. DQfD expects a model that has '
                "one dimension for each action, in this case {}.".format(
                    model.output, self.n_actions
                )
            )

        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            model = self._build_dueling_arch(model, dueling_type)
        self.model = model

        # multi-step learning parameter.
        self.n_step = n_step
        self.pretraining_steps = pretraining_steps
        self.pretraining = True
        # margin to add when action of agent != action of expert
        self.large_margin = large_margin
        # coefficient of supervised loss component of the loss function
        self.lam_2 = lam_2
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.reset_states()

        assert isinstance(
            self.memory, PartitionedMemory
        ), "DQfD needs a PartitionedMemory to store expert transitions without overwriting them."
        assert len(self.memory.observations) > 0, "Pre-load the memory with demonstration data."

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model; optimizer and loss choices are arbitrary
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer="sgd", loss="mse")
        self.model.compile(optimizer="sgd", loss="mse")

        # Compile model.
        if self.target_model_update < 1.0:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(
                self.target_model, self.model, self.target_model_update
            )
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def dqfd_error(args):
            (
                y_true,
                y_true_n,
                y_pred,
                importance_weights,
                agent_actions,
                large_margin,
                lam_2,
                mask,
            ) = args
            # Standard DQN loss
            j_dq = huber_loss(y_true, y_pred, self.delta_clip) * mask
            j_dq *= importance_weights
            j_dq = K.sum(j_dq, axis=-1)
            # N-step DQN loss
            j_n = huber_loss(y_true_n, y_pred, self.delta_clip) * mask
            j_n *= importance_weights
            j_n = K.sum(j_n, axis=-1)
            # Large margin supervised classification loss
            Q_a = y_pred * agent_actions
            Q_ae = y_pred * mask
            j_e = lam_2 * (Q_a + large_margin - Q_ae)
            j_e = K.sum(j_e, axis=-1)
            # in Keras, j_l2 from the paper is implemented as a part of
            # the network itself (using regularizers.l2)
            return j_dq + j_n + j_e

        y_pred = self.model.output
        y_true = Input(name="y_true", shape=(self.n_actions,))
        y_true_n = Input(name="y_true_n", shape=(self.n_actions,))
        mask = Input(name="mask", shape=(self.n_actions,))
        importance_weights = Input(name="importance_weights", shape=(self.n_actions,))
        agent_actions = Input(name="agent_actions", shape=(self.n_actions,))
        large_margin = Input(name="large-margin", shape=(self.n_actions,))
        lam_2 = Input(name="lam_2", shape=(self.n_actions,))
        loss_out = Lambda(dqfd_error, output_shape=(1,), name="loss")(
            [
                y_true,
                y_true_n,
                y_pred,
                importance_weights,
                agent_actions,
                large_margin,
                lam_2,
                mask,
            ]
        )
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(
            inputs=ins
            + [y_true, y_true_n, importance_weights, agent_actions, large_margin, lam_2, mask,],
            outputs=[loss_out, y_pred],
        )
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def fit(
        self,
        env,
        n_training_steps,
        action_repetition=1,
        callbacks=None,
        verbose=1,
        visualize=False,
        n_max_start_steps=0,
        start_step_policy=None,
        log_interval=10000,
        n_max_episode_steps=None,
    ):
        progbar = Progbar(self.pretraining_steps, interval=0.1)
        print("Pretraining for {} steps...".format(self.pretraining_steps))
        for step in range(self.pretraining_steps):
            self.backward(0, False)
            progbar.update(step)
        self.pretraining = False

        super(DQfDAgent, self).fit(
            env,
            n_training_steps,
            action_repetition,
            callbacks,
            verbose,
            visualize,
            n_max_start_steps,
            start_step_policy,
            log_interval,
            n_max_episode_steps,
        )

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0 and not self.pretraining:
            self.memory.append(
                self.recent_observation,
                self.recent_action,
                reward,
                terminal,
                training=self.training,
            )

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training and not self.pretraining:
            return metrics

        # Train the network on a single stochastic batch.
        if (self.step % self.train_interval == 0) or self.pretraining:
            # Calculations for current beta value based on a linear schedule.
            current_beta = self.memory.calculate_beta(self.step)
            # Sample from the memory.
            idxs = self.memory.sample_proportional(self.batch_size)
            experiences_n = self.memory.sample_by_idxs(
                idxs, self.batch_size, current_beta, self.n_step, self.gamma
            )
            experiences = self.memory.sample_by_idxs(idxs, self.batch_size, current_beta)

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            importance_weights = []
            for e in experiences[
                :-2
            ]:  # Prioritized Replay returns Experience tuple + weights and idxs.
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0.0 if e.terminal1 else 1.0)
            importance_weights = experiences[-2]

            # Get n-step versions. The state batch is the observation n steps
            # after the state0 batch (or the first terminal state, whichever is first).
            # The reward batch is the sum of discounted rewards between state0 and
            # the state in the state_batch_n. Terminal batch is used to eliminate
            # the target network's q values when the discounted rewards already extend
            # to the end of the episode.
            state_batch_n = []
            reward_batch_n = []
            terminal_batch_n = []
            for e in experiences_n[:-2]:
                state_batch_n.append(e.state1)
                reward_batch_n.append(e.reward)
                terminal_batch_n.append(0.0 if e.terminal1 else 1.0)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            state_batch_n = self.process_state_batch(state_batch_n)
            terminal1_batch = np.array(terminal1_batch)
            terminal_batch_n = np.array(terminal_batch_n)
            reward_batch = np.array(reward_batch)
            reward_batch_n = np.array(reward_batch_n)
            assert reward_batch.shape == (self.batch_size,)
            assert reward_batch_n.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                q_batch = self._calc_double_q_values(state1_batch)
                q_batch_n = self._calc_double_q_values(state_batch_n)
            else:
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.n_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
                # Repeat this process for the n-step state.
                target_q_values_n = self.target_model.predict_on_batch(state_batch_n)
                assert target_q_values_n.shape == (self.batch_size, self.n_actions)
                q_batch_n = np.max(target_q_values_n, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)
            assert q_batch_n.shape == (self.batch_size,)

            # Multi-step loss targets
            targets_n = np.zeros((self.batch_size, self.n_actions))
            masks = np.zeros((self.batch_size, self.n_actions))
            dummy_targets_n = np.zeros((self.batch_size,))
            discounted_reward_batch_n = (self.gamma ** self.n_step) * q_batch_n
            discounted_reward_batch_n *= terminal_batch_n
            assert discounted_reward_batch_n.shape == reward_batch_n.shape
            Rs_n = reward_batch_n + discounted_reward_batch_n
            for idx, (target, mask, R, action) in enumerate(
                zip(targets_n, masks, Rs_n, action_batch)
            ):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets_n[idx] = R
                mask[action] = 1.0  # enable loss for this specific action
            targets_n = np.array(targets_n).astype("float32")

            # Single-step loss targets
            targets = np.zeros((self.batch_size, self.n_actions))
            dummy_targets = np.zeros((self.batch_size,))
            discounted_reward_batch = self.gamma * q_batch
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
            targets = np.array(targets).astype("float32")
            masks = np.array(masks).astype("float32")

            # Make importance_weights the same shape as the other tensors that are
            # passed into the trainable model
            assert len(importance_weights) == self.batch_size
            importance_weights = np.array(importance_weights)
            importance_weights = np.vstack([importance_weights] * self.n_actions)
            importance_weights = np.reshape(importance_weights, (self.batch_size, self.n_actions))

            # we need the network to make its own decisions for each of the expert's
            # transitions (so we can compare)
            y_pred = self.model.predict_on_batch(state0_batch)
            agent_actions = np.argmax(y_pred, axis=1)
            assert agent_actions.shape == (self.batch_size,)
            # one-hot encode actions, gives the shape needed to pass into the model
            agent_actions = np.eye(self.n_actions)[agent_actions]
            expert_actions = masks
            # l is the large margin term, which skews loss function towards incorrect imitations
            large_margin = np.zeros_like(expert_actions, dtype="float32")
            # lambda_2 is used to eliminate supervised loss for self-generated transitions
            lam_2 = np.zeros_like(expert_actions, dtype="float32")

            # Here we are building the large margin term, which is a matrix
            # with a postiive entry where the agent and expert actions differ
            for i, idx in enumerate(idxs):
                if idx < self.memory.permanent_idx:
                    # this is an expert demonstration, enable supervised loss
                    lam_2[i, :] = self.lam_2
                    for j in range(agent_actions.shape[1]):
                        if agent_actions[i, j] == 1:
                            if expert_actions[i, j] != 1:
                                # if agent and expert had different predictions, increase l
                                large_margin[i, j] = self.large_margin

            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(
                ins
                + [
                    targets,
                    targets_n,
                    importance_weights,
                    agent_actions,
                    large_margin,
                    lam_2,
                    masks,
                ],
                [dummy_targets, targets],
            )

            assert len(idxs) == self.batch_size
            # Calculate new priorities.
            y_true = targets
            # Proportional method. Priorities are the abs TD error with a
            # small positive constant to keep them from being 0.
            # Boost for expert transitions is handled in memory.PartitionedMemory.update_priorities
            new_priorities = (abs(np.sum(y_true - y_pred, axis=-1))) + 0.001
            assert len(new_priorities) == self.batch_size
            self.memory.update_priorities(idxs, new_priorities)

            metrics = [
                metric for idx, metric in enumerate(metrics) if idx not in (1, 2)
            ]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    def get_config(self):
        config = super(DQfDAgent, self).get_config()
        config["enable_double_dqn"] = self.enable_double_dqn
        config["dueling_type"] = self.dueling_type
        config["enable_dueling_network"] = self.enable_dueling_network
        config["model"] = get_object_config(self.model)
        config["policy"] = get_object_config(self.policy)
        config["test_policy"] = get_object_config(self.test_policy)
        config["pretraining_steps"] = self.pretraining_steps
        config["n_step"] = self.n_step
        config["large_margin"] = self.large_margin
        if self.compiled:
            config["target_model"] = get_object_config(self.target_model)
        return config

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [
            name
            for idx, name in enumerate(self.trainable_model.metrics_names)
            if idx not in (1, 2)
        ]
        model_metrics = [name.replace(dummy_output_name + "_", "") for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
