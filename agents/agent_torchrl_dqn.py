"""Player based on a trained neural network"""
import json
# pylint: disable=wrong-import-order,invalid-name,import-error,missing-function-docstring
import logging
import time

import numpy as np
import torch.nn
import torch.optim
from torch import optim
from torchrl.data import CompositeSpec
from torchrl.data import ReplayBuffer, ListStorage
from torchrl.envs import RewardSum, StepCounter, TransformedEnv, GymLikeEnv
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.modules import MLP, QValueActor

from agents.player_interface import Player
from gym_env.enums import Action

autoplay = True  # play automatically if played against keras-rl

window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 50  # before training starts, should be higher than start steps
nb_steps = 100000
memory_limit = int(nb_steps / 2)
batch_size = 500  # items sampled from memory to train
enable_double_dqn = False

log = logging.getLogger(__name__)


class TorchRLDQN(Player):
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env: GymLikeEnv, device='cpu'):
        """initiate a deep Q agent"""

        nb_actions = self.env.action_space.n
        self.env = env
        self.model = DQN(env.observation_space, nb_actions).to(device)
        self.target_model = DQN(env.observation_space, nb_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(storage=ListStorage(max_size=memory_limit),
                                   batch_size=batch_size)
        q_net = trl.modules.QNetwork(self.model, self.target_model)
        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        # 這格應該最麻煩弄. todo:
        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        log.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        return action

    def train(self, env_name):
        """Train a model"""
        # initiate training loop
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        writer = SummaryWriter(log_dir=f'./Graph/{timestr}')

        self.dqn.fit(self.env, nb_max_start_steps=nb_max_start_steps, nb_steps=nb_steps, visualize=False, verbose=2,
                     start_step_policy=self.start_step_policy, callbacks=[tensorboard])

        # Save the architecture
        # dqn_json = self.model.to_json() // model comes from whole class definition

        # After training is done, we save the final weights.
        torch.save(self.model.state_dict(), 'dqn_{}_weights.pth'.format(env_name))

        # Finally, evaluate our algorithm for 5 episodes.
        # self.dqn.test(self.env, nb_episodes=5, visualize=False) todo: implement

    def load(self, env_name):
        """Load a model"""

        # Load the architecture
        with open('dqn_{}_json.json'.format(env_name), 'r') as architecture_json:
            dqn_json = json.load(architecture_json)

        self.model = model_from_json(dqn_json)
        self.model.load_weights('dqn_{}_weights.h5'.format(env_name))

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = TrumpPolicy()

        class CustomProcessor(Processor):  # pylint: disable=redefined-outer-name
            """The agent and the environment"""

            def process_state_batch(self, batch):
                """
                Given a state batch, I want to remove the second dimension, because it's
                useless and prevents me from feeding the tensor into my CNN
                """
                return np.squeeze(batch, axis=1)

            def process_info(self, info):
                processed_info = info['player_data']
                if 'stack' in processed_info:
                    processed_info = {'x': 1}
                return processed_info

        nb_actions = self.env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])  # pylint: disable=no-member

        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=render)

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action


class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")
        return action


class CustomProcessor(Processor):
    """The agent and the environment"""

    def __init__(self):
        """initizlie properties"""
        self.legal_moves_limit = None

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into cnn"""
        return np.squeeze(batch, axis=1)

    def process_info(self, info):
        if 'legal_moves' in info.keys():
            self.legal_moves_limit = info['legal_moves']
        else:
            self.legal_moves_limit = None
        return {'x': 1}  # on arrays allowed it seems

    def process_action(self, action):
        """Find nearest legal action"""
        if 'legal_moves_limit' in self.__dict__ and self.legal_moves_limit is not None:
            self.legal_moves_limit = [move.value for move in self.legal_moves_limit]
            if action not in self.legal_moves_limit:
                for i in range(5):
                    action += i
                    if action in self.legal_moves_limit:
                        break
                    action -= i * 2
                    if action in self.legal_moves_limit:
                        break
                    action += i

        return action


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="CartPole-v1", device="cpu"):
    env = GymEnv(env_name, device=device)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define Q-Value Module
    mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.ReLU,
        out_features=num_outputs,
        num_cells=[120, 84],
    )

    qvalue_module = QValueActor(
        module=mlp,
        spec=CompositeSpec(action=action_spec),
        in_keys=["observation"],
    )
    return qvalue_module


def make_dqn_model(env_name):
    proof_environment = make_env(env_name, device="cpu")
    qvalue_module = make_dqn_modules(proof_environment)
    del proof_environment
    return qvalue_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()
