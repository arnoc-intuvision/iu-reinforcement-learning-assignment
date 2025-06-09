import numpy as np
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard.writer import SummaryWriter

import ptan
from ptan.agent import DQNAgent, TargetNet
from ptan.actions import ArgmaxActionSelector
from ptan.experience import ExperienceFirstLast, ExperienceSourceFirstLast, PrioritizedReplayBuffer

import gymnasium as gym

from double_dqn_model_big import DoubleDQNModel

import logging as log

log.basicConfig(format="Line:%(lineno)d-%(funcName)s-%(levelname)s:  %(message)s")
log.getLogger().setLevel(log.INFO)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class DoubleDQNAgent:
    """
    A Double Deep Q-Network (DoubleDQN) agent for reinforcement learning.

    This agent implements the Double DQN algorithm, which improves upon the standard DQN
    by decoupling action selection and action evaluation to reduce overestimation bias.
    It utilizes a target network, prioritized experience replay, and noisy layers for exploration.
    """

    def __init__(self, env: gym.Env, 
                       discount_factor: float, 
                       td_n_steps_unroll: int,
                       initial_lr: float, 
                       final_lr_factor: float,
                       total_training_steps: int,
                       per_alpha: float,
                       per_beta: float,
                       per_beta_increment: float,
                       exp_batch_size: int, 
                       exp_buffer_size: int,
                       exp_min_buffer_samples: int,
                       target_model_sync_steps: int, 
                       episode_len: int,
                       stop_reward: float):
        """
        Initializes the DoubleDQNAgent.

        Args:
            env: The Gymnasium environment.
            discount_factor: The discount factor (gamma) for future rewards.
            td_n_steps_unroll: The number of steps for N-step TD learning.
            initial_lr: The initial learning rate for the optimizer.
            final_lr_factor: The factor by which the initial learning rate will be multiplied at the end of training.
            total_training_steps: The total number of training steps.
            per_alpha: The alpha parameter for prioritized experience replay (controls prioritization).
            per_beta: The initial beta parameter for prioritized experience replay (controls importance sampling).
            per_beta_increment: The increment for beta per training step.
            exp_batch_size: The size of the experience batch sampled from the replay buffer.
            exp_buffer_size: The maximum size of the prioritized experience replay buffer.
            exp_min_buffer_samples: The minimum number of samples in the buffer before training starts.
            target_model_sync_steps: The number of steps after which the target network is synchronized with the main network.
            episode_len: The length of each training episode.
            stop_reward: The reward threshold at which training can be stopped.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.td_n_steps_unroll = td_n_steps_unroll
        self.initial_lr = initial_lr
        self.final_lr_factor = final_lr_factor
        self.total_training_steps = total_training_steps
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        self.current_beta = per_beta
        self.exp_batch_size = exp_batch_size
        self.exp_buffer_size = exp_buffer_size
        self.exp_min_buffer_samples = exp_min_buffer_samples
        self.target_model_sync_steps = target_model_sync_steps
        self.episode_len = episode_len
        self.stop_reward = stop_reward
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.total_steps = 0
        
        self.dqn_net = DoubleDQNModel(input_shape=self.env.observation_space.shape[0],
                                      n_actions=self.env.action_space.n).to(self.device)

        self.tgt_dqn_net = TargetNet(self.dqn_net)

        # Initialize the DQN Model's Optimizer with learning rate scheduler
        self.dqn_net_optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.initial_lr)
        self.lr_lambda = lambda step: max(self.final_lr_factor, 1.0 - (1.0 - self.final_lr_factor) * ((step + self.exp_min_buffer_samples) / self.total_training_steps))
        self.lr_scheduler = LambdaLR(self.dqn_net_optimizer, lr_lambda=self.lr_lambda)

        # Argmax Action Selector
        self.action_selector = ArgmaxActionSelector()

        # The agent uses the DQN Model to convert observations into actions using Argmax action selection for exploration
        self.agent = DQNAgent(model=self.dqn_net, preprocessor=ptan.agent.float32_preprocessor, action_selector=self.action_selector, device=self.device)

        # Uses the agent to interact with the environment to obtain experiences (s, a, r, s')
        self.exp_source = ExperienceSourceFirstLast(self.env, self.agent, gamma=self.discount_factor, env_seed=SEED, steps_count=self.td_n_steps_unroll)

        # Stores experiences into the prioritzed experience replay buffer, and allows for prioritized sampling from the buffer
        self.exp_buffer = PrioritizedReplayBuffer(self.exp_source, buffer_size=self.exp_buffer_size, alpha=self.per_alpha)

        # Register Tensor Board Writer
        self.writer = SummaryWriter(comment="DoubleDQNAgent")

        # Display the Double-Q-Learning Model Architecture
        self.display_dqn_net()


    def display_dqn_net(self):
        """
        Displays the Double DQN model architecture and the compute device being used.
        """
        log.info(f"\nTorch Compute Device -> {self.device}\n")

        log.info("Double DQN Model Architecture -> ")
        print(self.dqn_net)


    def gather_agent_env_experiences(self) -> tuple:
        """
        Gathers experiences by allowing the agent to interact with the environment.

        The agent populates the experience buffer until it has at least `exp_min_buffer_samples`.
        Then, it samples a batch of experiences from the buffer using prioritized sampling.

        Returns:
            A tuple containing:
                - exp_batch: A batch of experiences.
                - indices: The indices of the sampled experiences in the buffer.
                - weights: The importance sampling weights for the sampled experiences.
        """
        for idx in range(self.exp_min_buffer_samples):

            # 1) Take the current environment observation as input and select an action 
            # 2) Execute action against the environment env.step(action)
            # 3) Obtain a new experience from the environment => (s, a, r, s')
            # 4) Store the experience in the replay buffer
            # 5) Reset environment if terminal state was reached
            
            self.exp_buffer.populate(1)
            
            self.total_steps += 1
            
            # Decay epsilon based on the total number of iterations
            # self.epsilon_tracker.frame(self.total_steps)
            
            # At the start, populate at least exp_min_buffer_size interactions with the environment
            if len(self.exp_buffer) < self.exp_min_buffer_samples:
                continue
            else:
                break

        # Update beta parameter for prioritized sampling
        self.current_beta = min(1.0, self.current_beta + self.per_beta_increment)
    
        # Sample a batch with priorities using the current beta value
        exp_batch, indices, weights = self.exp_buffer.sample(self.exp_batch_size, self.current_beta)
    
        return exp_batch, indices, weights


    def experience_batch_to_tensors(self, exp_batch: tt.List[ExperienceFirstLast]):
        """
        Converts a batch of experiences into PyTorch tensors.

        Args:
            exp_batch: A list of ExperienceFirstLast objects.

        Returns:
            A tuple containing:
                - states_t: Tensor of states.
                - actions_t: Tensor of actions.
                - rewards_t: Tensor of rewards.
                - last_states_t: Tensor of next states.
                - dones_t: Tensor indicating if the episode ended.
        """
        states, actions, rewards, last_states, dones = [], [], [], [], []
        
        for e in exp_batch:
            
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            dones.append(e.last_state is None)

            if e.last_state is None:
                last_states.append(e.state)
            else:
                last_states.append(e.last_state)
                
        # Convert the experience fields to pytorch tensors with the correct data types
        states_t = torch.as_tensor(np.asarray(states)).float().to(self.device)
        actions_t = torch.as_tensor(np.asarray(actions)).long().to(self.device) 
        rewards_t = torch.as_tensor(np.asarray(rewards)).float().to(self.device)
        last_states_t = torch.as_tensor(np.asarray(last_states)).float().to(self.device)
        dones_t = torch.as_tensor(np.asarray(dones)).bool().to(self.device)
        
        return states_t, actions_t, rewards_t, last_states_t, dones_t

    
    def calc_double_dqn_loss(self, exp_batch, weights) -> tuple:
        """
        Calculates the Double DQN loss.

        This method implements the core Double DQN loss calculation, using the main network
        to select the best action in the next state and the target network to evaluate that action.
        It also incorporates importance sampling weights from the prioritized replay buffer.

        Args:
            exp_batch: A batch of experiences.
            weights: Importance sampling weights for the batch.

        Returns:
            A tuple containing:
                - loss: The calculated weighted mean squared error loss.
                - td_errors: The temporal difference errors for each experience in the batch.
        """
        states_v, actions_v, rewards_v, next_states_v, done_mask = self.experience_batch_to_tensors(exp_batch)
        
        # Convert weights to tensor
        weights_v = torch.as_tensor(np.asarray(weights)).float().to(self.device)
        
        # Q(s, a) <- model(current_states)
        actions_v = actions_v.unsqueeze(-1)
        state_action_values = self.dqn_net(states_v).gather(1, actions_v).squeeze(-1)

        next_state_action_values = None
        with torch.no_grad():
            # Q_max(s', a) <- target_model(next_states)
            next_state_action_max_indices = self.dqn_net(next_states_v).max(1)[1]
            next_state_action_max_indices = next_state_action_max_indices.unsqueeze(-1)
            next_state_action_values = self.tgt_dqn_net.target_model(next_states_v).gather(1, next_state_action_max_indices).squeeze(-1)
            next_state_action_values[done_mask] = 0.0

        # Q_target(s') <- r + [gamma * Q_max(s', a)]
        target_state_action_values = next_state_action_values.detach() * (self.discount_factor**self.td_n_steps_unroll) + rewards_v

        # Calculate TD errors for updating priorities
        td_errors = torch.abs(state_action_values - target_state_action_values).detach().cpu().numpy()
        
        # Calculate weighted MSE loss
        loss = (state_action_values - target_state_action_values) ** 2
        weighted_loss = loss * weights_v
        return weighted_loss.mean(), td_errors

    
    def calculate_total_episode_reward(self):
        """
        Calculates the total reward for the most recent episode stored in the experience buffer.

        Returns:
            The total reward for the episode.
        """
        total_reward = 0.0
        
        for e in list(reversed(self.exp_buffer.buffer)):

            if e.last_state is None:
                break

            total_reward += e.reward
            
        return total_reward
        
    
    def learn(self):
        """
        The main training loop for the Double DQN agent.

        This method orchestrates the agent's interaction with the environment,
        experience gathering, loss calculation, model optimization, and target network updates.
        It also logs training progress to TensorBoard and saves the best model.
        """

        episode_number = 0
        best_episode_reward = -999_999
        episode_rewards = []

        while True:

            # Let the DQN agent interact with the environment, and gather a batch of experiences
            exp_batch, indices, weights = self.gather_agent_env_experiences()

            # Reset the dqn_net model's optimizer gradients to zero
            self.dqn_net_optimizer.zero_grad()

            # Calculate the Double DQN Loss with importance sampling weights
            loss_t, td_errors = self.calc_double_dqn_loss(exp_batch, weights)
            self.writer.add_scalar("model_loss", loss_t, self.total_steps)
            
            # Update priorities in the buffer
            priorities = [max(float(err), 1e-5) for err in td_errors]
            self.exp_buffer.update_priorities(indices, priorities)

            # Log the current beta value
            self.writer.add_scalar("per_beta", self.current_beta, self.total_steps)

            # Calculate the gradient of the loss function (loss_t) with respect to the dqn_net model's weights
            loss_t.backward()
            
            # Update the dqn_net model's weights
            self.dqn_net_optimizer.step()

            # Step the learning rate scheduler
            self.lr_scheduler.step()

            current_lr_list = self.lr_scheduler.get_last_lr()
            if current_lr_list:
                current_lr = current_lr_list[0]
                self.writer.add_scalar("current_lr", current_lr, self.total_steps)
            else:
                current_lr = self.initial_lr
                self.writer.add_scalar("current_lr", current_lr, self.total_steps)
            
            # Reset the dqn_net model's noisy linear layers
            self.dqn_net.reset_noise()

            # Check if the tgt_dqn_net model's weights needs to be synced with the dqn_net model's weights
            if (self.total_steps % self.target_model_sync_steps) == 0:
                print(f"    [{self.total_steps}] -> Sync target model weights !")
                self.tgt_dqn_net.sync()

            # Check if the episode length has been reached, and calculate the total episode reward
            if (self.total_steps % self.episode_len) == 0:
                
                total_episode_reward = self.calculate_total_episode_reward()
                episode_rewards.append(total_episode_reward)
                
                self.writer.add_scalar("total_episode_reward", total_episode_reward, episode_number)
                
                mean_episode_reward = 0.0
                if (episode_number % 10) == 0:
                    
                    mean_episode_reward = np.mean(episode_rewards)
                    episode_rewards = []
                    
                    self.writer.add_scalar("mean_episode_reward", mean_episode_reward, episode_number)
                    
                    if best_episode_reward < mean_episode_reward:
                        
                        print(f"""
                        New Best Episode Reward Metrics:
                            Episode Number: {episode_number}
                            Step Number: {self.total_steps}
                            Best Episode Reward: Prev:{best_episode_reward: .2f} -> New:{mean_episode_reward: .2f}
                        """)

                        best_episode_reward = mean_episode_reward
                        torch.save(self.dqn_net.state_dict(), "./model_checkpoints/double_dqn_model_weights.pth")

                if (self.total_steps > self.total_training_steps) or (mean_episode_reward > self.stop_reward):
                    self.total_steps = 0
                    break

                episode_number += 1



