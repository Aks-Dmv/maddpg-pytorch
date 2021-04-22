from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork, DNRI_Encoder, DNRI_MLP_Decoder, MLP_Qnet
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class RL_DNRIAgent(object):
    """
    General class for RL_DNRIAgent agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_vars, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.encoder = DNRI_Encoder(num_in_pol, num_out_pol, num_vars, 
                            hidden_dim)
        self.decoder = DNRI_MLP_Decoder(num_in_pol, num_out_pol, num_vars, 
                            hidden_dim, discrete_action=discrete_action)
        
        self.critic = MLP_Qnet(num_in_pol + hidden_dim, num_out_pol, hidden_dim)
        self.target_critic = MLP_Qnet(num_in_pol + hidden_dim, num_out_pol, hidden_dim)

        self.target_critic.eval()

        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam( list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, enc_hid, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        prior_logits, new_enc_hid = self.encoder(obs, enc_hid)
        prior_logits = gumbel_softmax(prior_logits, hard=False)
        
        pi_action, logp_pi, decoder_hidden_state = self.decoder(obs, prior_logits)
        if explore:
            expl_noise = Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            if pi_action.is_cuda:
                expl_noise = expl_noise.cuda()
            pi_action += expl_noise
        pi_action = pi_action.clamp(-1, 1)
        return pi_action, logp_pi, new_enc_hid, decoder_hidden_state

    def get_params(self):
        return {'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.encoder.load_state_dict(params['encoder'])
        self.decoder.load_state_dict(params['decoder'])
        self.critic.load_state_dict(params['critic'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])