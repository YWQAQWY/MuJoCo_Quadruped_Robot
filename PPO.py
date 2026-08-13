import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim = 1)

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
    return torch.tensor(advantage_list, dtype=torch.float)

class PPO():
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epoch, eps, actor_gamma, critic_gamma, device):
        self.actor_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_net = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr= actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(),lr=critic_lr)
        self.actor_gamma = actor_gamma
        self.critic_gamma = critic_gamma
        self.lmbda = lmbda # GAE λ 参数
        self.epoch = epoch #一条序列训练epoch轮
        self.eps =eps #PPO截断范围参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.critic_gamma * self.critic_net(next_states)*(1-dones)
        td_delta = td_target - self.critic_net(states)

        advantadge = compute_advantage(self.actor_gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor_net(states).gather(1,actions)).detach()

        for _ in range(self.epoch):
            log_probs = torch.log(self.actor_net(states).gather(1,actions))
            ratio = torch.exp(log_probs - old_log_probs) #π_new / π_old
            surr1 = ratio * advantadge
            surr2 = torch.clamp(ratio,1-self.eps,1+self.eps) * advantadge #PPO-裁断
            actor_loss = torch.mean(-torch.min(surr1,surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic_net(states),td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
