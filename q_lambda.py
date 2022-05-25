from typing import Dict
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag


class Network(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float
    ) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ReplayBuffer:
    def __init__(
        self,
        input_size: int,
        buffer_size: int,
        batch_size: int
    ) -> None:
        self.states_memory = np.zeros((buffer_size, input_size), dtype = np.float32)
        self.actions_memory = np.zeros(buffer_size, dtype = np.longlong)
        self.rewards_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_states_memory = np.zeros((buffer_size, input_size), dtype = np.float32)
        self.terminal_states_memory = np.zeros(buffer_size, dtype = np.bool8)
        self.cur_size, self.ptr = 0, 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.states_memory[self.ptr] = state
        self.actions_memory[self.ptr] = action
        self.rewards_memory[self.ptr] = reward
        self.next_states_memory[self.ptr] = next_state
        self.terminal_states_memory[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
    
    def sample(self) -> Dict[str, torch.Tensor]:
        idxs = np.random.choice(self.cur_size, self.batch_size, False)
        return dict(
            states = torch.from_numpy(self.states_memory[idxs]),
            actions = torch.from_numpy(self.actions_memory[idxs]),
            rewards = torch.from_numpy(self.rewards_memory[idxs]),
            next_states = torch.from_numpy(self.next_states_memory[idxs]),
            terminal_states = torch.from_numpy(self.terminal_states_memory[idxs])
        )

    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class EpsilonController:
    def __init__(self, init_eps: float, min_eps: float, epsilon_decay: float) -> None:
        self.min_max_diff = init_eps - min_eps
        self.min_eps = min_eps
        self.epsilon_decay = epsilon_decay
    
    def get_epsilon(self, steps: int) -> float:
        return self.min_eps + (self.min_max_diff) * np.exp(-1.0 * steps / self.epsilon_decay)


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float,
        gamma: float,
        lamb: float,
        buffer_size: int,
        batch_size: int,
        init_eps: float,
        min_eps: float,
        epsilon_decay: float
    ) -> None:
        self.network = Network(input_size, output_size, learning_rate)
        self.replaybuffer = ReplayBuffer(input_size, buffer_size, batch_size)
        self.epsiloncontroller = EpsilonController(init_eps, min_eps, epsilon_decay)
        
        self.batch_index = np.arange(batch_size, dtype = np.longlong)
        self.output_size = output_size
        self.gamma = gamma
        self.lamb = lamb
        self.trace = dict()
        self.reset_trace()
    
    def choose_action_train(self, state: np.ndarray, steps: int) -> int:
        epsilon = self.epsiloncontroller.get_epsilon(steps)
        if np.random.random() > epsilon:
            state = torch.as_tensor(state, dtype = torch.float32)
            return self.network.forward(state).argmax().item()
        else:
            return np.random.choice(self.output_size)
    
    def choose_action_test(self, state: np.ndarray) -> int:
        state = torch.as_tensor(state, dtype = torch.float32)
        return self.network.forward(state).argmax().item()
    
    def reset_trace(self) -> None:
        for idx, param_set in enumerate(self.network.parameters()):
            self.trace[idx] = torch.zeros(param_set.data.shape)
    
    def train(self) -> float:
        states, actions, rewards, next_states, terminal_states = self.replaybuffer.sample().values()
        q_pred = self.network.forward(states)[self.batch_index, actions]
        q_next = self.network.forward(next_states).max(1)[0]
        q_target = rewards + self.gamma * q_next * ~terminal_states
        loss = self.network.loss(q_target.detach(), q_pred)
        self.network.optimizer.zero_grad()
        eval_gradients = ag.grad(loss, self.network.parameters())
        for idx, param_set in enumerate(self.network.parameters()):
            self.trace[idx] = self.gamma * self.lamb * self.trace[idx] + eval_gradients[idx]
            param_set.grad = loss * self.trace[idx]
        self.network.optimizer.step()
        return loss.item()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(
        *env.observation_space.shape, env.action_space.n, 0.001,
        0.99, 0.5, 10000, 32, 1.0, 0.001, 500.0
    )
    episodes = 150

    steps = 0
    for i in range(episodes):
        state = env.reset()
        done = False
        loss = 0
        score = 0
        while not done:
            action = agent.choose_action_train(state, steps)
            next_state, reward, done, _ = env.step(action)
            if agent.replaybuffer.is_ready():
                loss = agent.train()
            agent.replaybuffer.store(state, action, reward, next_state, done)
            score += reward
            steps += 1
            state = next_state
        agent.reset_trace()
        print(f"Iteration: {i + 1}, Score: {score}, Loss: {loss}")
