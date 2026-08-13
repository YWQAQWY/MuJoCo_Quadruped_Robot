import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym

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


# =====================
# 训练
# =====================

def train_ppo(render=False):
    """
    在 CartPole-v1 上训练 PPO

    参数:
        render: True=每一步都实时渲染 (弹出窗口, 可以看小车和杆子)
    """

    # ---------- 超参数 ----------
    num_episodes = 500
    hidden_dim = 128
    actor_lr = 1e-3
    critic_lr = 1e-2
    actor_gamma = 0.98
    critic_gamma = 0.98
    lmbda = 0.95              # GAE λ
    epoch = 10                # 同一条轨迹重复训练轮数
    eps = 0.2                 # PPO clip 范围 ε
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_render_mode = "human" if render else "rgb_array"

    print(f"[Device] {device}")
    print(f"[Env] CartPole-v1")
    print(f"[Algorithm] PPO (Proximal Policy Optimization)")
    print(f"[Net] Actor: 4->{hidden_dim}->2(Softmax)  |  Critic: 4->{hidden_dim}->1")
    print(f"[Train] {num_episodes} episodes, gamma={actor_gamma}, lambda={lmbda}")
    print(f"[PPO] epochs={epoch}, eps={eps}")
    if render:
        print(f"[Render] LIVE — 每一步都实时渲染!")

    # ---------- 初始化 ----------
    np.random.seed(0)
    torch.manual_seed(0)

    env = gym.make("CartPole-v1", render_mode=train_render_mode)
    state_dim = env.observation_space.shape[0]    # 4
    action_dim = env.action_space.n                # 2

    agent = PPO(state_dim, hidden_dim, action_dim,
                actor_lr, critic_lr, lmbda, epoch, eps,
                actor_gamma, critic_gamma, device)

    # ---------- 训练循环 ----------
    return_list = []

    for i_episode in range(num_episodes):
        episode_return = 0
        transition_dict = {
            'states': [], 'actions': [], 'next_states': [],
            'rewards': [], 'dones': []
        }

        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.take_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transition_dict['states'].append(obs)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_obs)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            obs = next_obs
            episode_return += reward

        # 一局结束, PPO 更新 (对同一批数据重复训练 epochs 轮)
        agent.update(transition_dict)
        return_list.append(episode_return)

        # ---- 打印进度 ----
        if (i_episode + 1) % 50 == 0:
            avg = np.mean(return_list[-50:])
            print(f"[Ep {i_episode+1:4d}/{num_episodes}] "
                  f"avg(last 50): {avg:6.1f}")

    env.close()

    # ---------- 结果 ----------
    final_avg = np.mean(return_list[-100:])
    print(f"\n{'='*50}")
    print(f"Training done! {num_episodes} episodes")
    print(f"Avg over last 100 episodes: {final_avg:.1f}")
    print(f"Max single-episode return: {np.max(return_list):.0f}")
    print(f"{'='*50}")

    return agent, return_list


# =====================
# CartPole 渲染播放
# =====================

def play_episode(agent, seed=None, max_steps=500):
    """
    弹出一个窗口，实时渲染 agent 玩一局 CartPole。
    你可以在窗口里直接看到小车和杆子的运动。
    """
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset(seed=seed)
    total_reward = 0

    for _ in range(max_steps):
        env.render()
        action = agent.take_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()
    return total_reward


def play_multiple(agent, episodes=5):
    """连续渲染多局"""
    print(f"\n{'='*50}")
    print(f"  CartPole Live Render - {episodes} episodes")
    print( "  Watch the cart-pole in the popup window!")
    print(f"{'='*50}")

    scores = []
    for ep in range(episodes):
        input(f">>> Episode {ep+1}/{episodes} ready. Press Enter to start...")
        score = play_episode(agent, seed=ep)
        scores.append(score)
        print(f"    Episode {ep+1}: score = {score}")

    print(f"\n  Avg: {np.mean(scores):.1f}  |  "
          f"Max: {np.max(scores):.0f}  |  Min: {np.min(scores):.0f}")