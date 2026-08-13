import numpy as np
import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        # 高斯策略的可学习 log σ（状态无关，每个动作维度一个）
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 输出动作均值 μ（连续高斯策略，不再用 softmax）

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

class PPO():
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epoch, eps, batch_size, actor_gamma, critic_gamma, device, entropy_coef=0.01):
        self.actor_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_net = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr= actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(),lr=critic_lr)
        self.actor_gamma = actor_gamma
        self.critic_gamma = critic_gamma
        self.lmbda = lmbda # GAE λ 参数
        self.epoch = epoch #一条序列训练epoch轮
        self.eps =eps #PPO截断范围参数
        self.batch_size = batch_size #每个epoch内小批量更新的批量大小
        self.entropy_coef = entropy_coef #熵正则系数：防止高斯 σ 塌缩、保持探索
        self.device = device

    def take_action(self, state, deterministic=False):
        """高斯采样动作。deterministic=True 时返回均值（评估/回放用）。"""
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            mu = self.actor_net(state)
            std = self.actor_net.log_std.exp()
            if deterministic:
                action = mu
            else:
                action = torch.normal(mu, std)  # 高斯采样，σ 即连续版的探索
            return action.squeeze(0).cpu().numpy()

    def update(self,transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device) # [N, action_dim]
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.critic_gamma * self.critic_net(next_states)*(1-dones)
        td_delta = td_target - self.critic_net(states)

        advantadge = compute_advantage(self.actor_gamma, self.lmbda, td_delta.cpu()).to(self.device)

        # 旧策略对数概率（更新前的网络，每轮 epoch 保持不变）
        mu = self.actor_net(states)
        std = self.actor_net.log_std.exp()
        old_dist = torch.distributions.Normal(mu, std)
        old_log_probs = old_dist.log_prob(actions).sum(dim=-1).detach()

        n = states.shape[0]
        for _ in range(self.epoch):
            index = torch.randperm(n, device=self.device) #每个epoch先随机打乱数据
            for start in range(0, n, self.batch_size):
                idx = index[start : start + self.batch_size] #切成小批量
                mu_b = self.actor_net(states[idx])
                std_b = self.actor_net.log_std.exp()
                dist_b = torch.distributions.Normal(mu_b, std_b)
                log_probs = dist_b.log_prob(actions[idx]).sum(dim=-1)
                ratio = torch.exp(log_probs - old_log_probs[idx]) #π_new / π_old
                surr1 = ratio * advantadge[idx]
                surr2 = torch.clamp(ratio,1-self.eps,1+self.eps) * advantadge[idx] #PPO-裁断
                actor_loss = torch.mean(-torch.min(surr1,surr2))
                # 熵正则：奖励策略保持探索，防止 σ 快速塌缩到 0
                entropy = dist_b.entropy().sum(dim=-1).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy
                critic_loss = torch.mean(F.mse_loss(self.critic_net(states[idx]),td_target[idx].detach()))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
