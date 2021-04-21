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
    def __init__(self, agent_init_params, alg_types,
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
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        # because all agents are identical
        # self.agents = RL_DNRIAgent(agent_init_params[0]["num_in_pol"], agent_init_params[0]["num_out_pol"], 
        #                 agent_init_params[0]["num_vars"], hidden_dim=hidden_dim, lr=lr, discrete_action=discrete_action)

        self.agents = RL_DNRIAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **(agent_init_params[0]))

        self.hidden_dim = hidden_dim
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.alpha = 0.4
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

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
        pi_action, _, new_enc_hid, _ = self.agents.step(torch_obs, enc_hid)
        pi_action = pi_action.transpose(0, 1).contiguous() # back to [agent, batch, hidden]
        return pi_action, new_enc_hid

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
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
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()
        curr_agent.critic_optimizer.zero_grad()

        pi_action, logp_pi, _, dec_hidden = curr_agent.step(obs, hid_enc)
        
        q1_pi, q2_pi = curr_agent.critic(dec_hidden.detach(), pi_action)
        q_pi = torch.min(q1_pi, q2_pi).sum(dim=-1)

        pol_loss = (self.alpha * logp_pi - q_pi ).mean()
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.encoder)
            average_gradients(curr_agent.decoder)
        torch.nn.utils.clip_grad_norm_(curr_agent.encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(curr_agent.decoder.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.agents.target_critic, self.agents.critic, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        self.agents.encoder.train()
        self.agents.decoder.train()
        self.agents.critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.agents.encoder = fn(self.agents.encoder)
            self.agents.decoder = fn(self.agents.decoder)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.agents.critic = fn(self.agents.critic)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            self.agents.target_critic = fn(self.agents.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        self.agents.encoder.eval()
        self.agents.decoder.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            self.agents.encoder = fn(self.agents.encoder)
            self.agents.decoder = fn(self.agents.decoder)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': self.agents.get_params()}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_vars': len(alg_types)})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.agents.load_params(save_dict['agent_params'])
        return instance