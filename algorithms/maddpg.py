import torch
import torch.nn.functional as F
from torch.autograd import Variable
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent, RL_DNRIAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, nagents,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = nagents
        # because all agents are identical
        # self.agents = RL_DNRIAgent(agent_init_params[0]["num_in_pol"], agent_init_params[0]["num_out_pol"], 
        #                 agent_init_params[0]["num_vars"], hidden_dim=hidden_dim, lr=lr, discrete_action=discrete_action)

        self.agents = RL_DNRIAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **agent_init_params)

        self.hidden_dim = hidden_dim
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.alpha = 0.4
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.niter = 0
        self.prep_training(device='gpu')

    @property
    def policies(self):
        return [self.agents.encoder, self.agents.decoder]

    @property
    def target_policies(self):
        return [self.agents.encoder, self.agents.decoder]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        return 0

    def reset_noise(self):
        self.agents.reset_noise()

    def step(self, observations, enc_hid, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        torch_obs = torch.stack(observations, dim=0).transpose(0, 1).contiguous()
        torch_obs = Variable(torch_obs, requires_grad=False)
        if self.pol_dev == 'gpu':
            torch_obs = torch_obs.cuda()
        pi_action, _, new_enc_hid, _ = self.agents.step(torch_obs, enc_hid)
        pi_action = pi_action.transpose(0, 1).contiguous() # back to [agent, batch, hidden]
        return pi_action, new_enc_hid

    def update(self, sample):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
        """
        obs, hid_enc0, hid_enc1, acs, rews, next_obs, next_hid_enc0, next_hid_enc1, dones = sample
        obs = torch.stack(obs, dim=0).transpose(0, 1).contiguous()
        hid_enc0 = torch.stack(hid_enc0, dim=0).transpose(0, 1).contiguous().view(obs.shape[0],-1, self.hidden_dim)
        hid_enc1 = torch.stack(hid_enc1, dim=0).transpose(0, 1).contiguous().view(obs.shape[0],-1, self.hidden_dim)
        acs = torch.stack(acs, dim=0).transpose(0, 1).contiguous()
        rews = torch.stack(rews, dim=0).transpose(0, 1).contiguous()
        next_obs = torch.stack(next_obs, dim=0).transpose(0, 1).contiguous()
        next_hid_enc0 = torch.stack(next_hid_enc0, dim=0).transpose(0, 1).contiguous().view(obs.shape[0],-1, self.hidden_dim)
        next_hid_enc1 = torch.stack(next_hid_enc1, dim=0).transpose(0, 1).contiguous().view(obs.shape[0],-1, self.hidden_dim)
        dones = torch.stack(dones, dim=0).transpose(0, 1).contiguous()

        hid_enc = (hid_enc0, hid_enc1)
        next_hid_enc = (next_hid_enc0, next_hid_enc1)

        curr_agent = self.agents
        if self.pol_dev == 'gpu':
            obs = obs.cuda()
            rews = rews.cuda()
            next_obs = next_obs.cuda()
            dones = dones.cuda()

        curr_agent.policy_optimizer.zero_grad()
        curr_agent.critic_optimizer.zero_grad()
        pi_action2, logp_pi, _, trgt_vf_in = curr_agent.step(next_obs, next_hid_enc)
        
        q1_pi_targ, q2_pi_targ = curr_agent.target_critic(trgt_vf_in.detach(), pi_action2.detach())
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).sum(dim=-1)
        target_value = (rews + self.gamma *
                        ( q_pi_targ - self.alpha * logp_pi.detach()) *
                        (1 - dones))


        pi_action1, _, _, vf_in = curr_agent.step(obs, hid_enc)
        actual_value1, actual_value2  = curr_agent.critic(vf_in.detach(), pi_action1.detach())
        vf_loss = MSELoss(actual_value1.sum(dim=-1), target_value.detach()) + MSELoss(actual_value2.sum(dim=-1), target_value.detach())
        vf_loss.backward()
        print(vf_loss,"vf")
        
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()
        curr_agent.critic_optimizer.zero_grad()

        pi_action, logp_pi, _, dec_hidden = curr_agent.step(obs, hid_enc)
        
        q1_pi, q2_pi = curr_agent.critic(dec_hidden.detach(), pi_action)
        q_pi = torch.min(q1_pi, q2_pi).sum(dim=-1)

        pol_loss = (self.alpha * logp_pi - q_pi ).mean()
        pol_loss.backward()
        print(pol_loss,"pol")
        curr_agent.policy_optimizer.step()

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.agents.target_critic, self.agents.critic, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        if device == 'gpu':
            self.agents.encoder = self.agents.encoder.cuda()
            self.agents.decoder = self.agents.decoder.cuda()
            self.agents.critic = self.agents.critic.cuda()
            self.agents.target_critic = self.agents.target_critic.cuda()
            self.pol_dev = device
        self.agents.encoder.train()
        self.agents.decoder.train()
        self.agents.critic.train()
        self.agents.target_critic.eval()

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': self.agents.get_params()}
        torch.save(save_dict, filename)
