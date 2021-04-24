import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class PolicyNetwork(nn.Module):
    """
    MLP Policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                norm_in=False, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(PolicyNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_layer = nn.Linear(hidden_dim, out_dim)
        self.log_std_layer = nn.Linear(hidden_dim, out_dim)

        self.nonlin = nonlin
        if not discrete_action:
            # initialize small to prevent saturation
            self.mu_layer.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))

        mu = self.mu_layer(h2)
        log_std = self.log_std_layer(h2)
        log_std = torch.clamp(log_std, -3, 1)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if self.training:
            pi_action = pi_distribution.rsample()
        else:
            pi_action = mu

        logp_pi = pi_distribution.log_prob(pi_action).sum(dim=-1)
        log_2_torch = torch.log(torch.Tensor([2.]))
        if pi_action.is_cuda:
            log_2_torch = log_2_torch.cuda()
        logp_pi -= (2*(log_2_torch - pi_action - F.softplus(-2*pi_action))).sum(dim=-1)
        
        pi_action = self.out_fn(pi_action)
        return pi_action, logp_pi


class GNNNetwork(nn.Module):
    """
    MLP GNN network
    """
    def __init__(self, input_dim, out_dim, num_vars, edge_types = 2, hidden_dim=64, nonlin=F.relu,
                norm_in=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(GNNNetwork, self).__init__()
        self.num_edges = edge_types
        n_hid = hidden_dim
        msg_hid = hidden_dim
        msg_out = hidden_dim
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges)), requires_grad=False)

        self.dropout_prob = 0.3
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * input_dim, msg_hid) for _ in range(self.num_edges)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(self.num_edges)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = True

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.nonlin = nonlin

    def forward(self, inputs, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        
        # Node2edge
        receivers = inputs[:, self.recv_edges, :]
        senders = inputs[:, self.send_edges, :]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                            self.msg_out_shape)

        if pre_msg.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0
        if self.training:
            p = self.dropout_prob
        else:
            p = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = self.nonlin(self.msg_fc1[i](self.in_fn(pre_msg)))
            msg = F.dropout(msg, p=p)
            msg = self.nonlin(self.msg_fc2[i](msg))
            msg = msg * edges[:, :, i:i + 1]
            all_msgs += msg
        
        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(self.edge2node_mat).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)
        return aug_inputs

class DNRI_MLP_Decoder(nn.Module):
    """
    DNRI_MLP_Decoder network
    """
    def __init__(self, input_dim, out_dim, num_vars, hidden_dim=64, nonlin=F.relu,
                norm_in=False, discrete_action=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(DNRI_MLP_Decoder, self).__init__()
        self.num_edges = 2

        self.gnn_net = GNNNetwork(input_dim, hidden_dim, num_vars, self.num_edges, hidden_dim, nonlin,
                norm_in)
        self.policy_net = PolicyNetwork(input_dim + hidden_dim, out_dim, hidden_dim, nonlin,
                norm_in, discrete_action)

    def forward(self, inputs, edges):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        
        aug_inputs = self.gnn_net(inputs, edges)
        pi_action, logp_pi = self.policy_net(aug_inputs)
        return pi_action, logp_pi, aug_inputs

class MLP_Qnet(nn.Module):
    """
    MLP_Qnet network
    """
    def __init__(self, input_dim, act_dim, hidden_dim=64):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLP_Qnet, self).__init__()
        
        self.critic1 = MLPNetwork(input_dim + act_dim, 1,
                        hidden_dim=hidden_dim,
                        constrain_out=False)
        self.critic2 = MLPNetwork(input_dim + act_dim, 1,
                        hidden_dim=hidden_dim,
                        constrain_out=False)

    def forward(self, inputs, pi_act):

        # single_timestep_inputs has shape
        # [batch_size, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]
        
        q1 = self.critic1(torch.cat([inputs, pi_act], dim=-1))
        q2 = self.critic2(torch.cat([inputs, pi_act], dim=-1))
        return q1, q2

class DNRI_Encoder(nn.Module):
    """
    DNRI_Encoder network
    """
    def __init__(self, input_dim, out_dim, num_vars, hidden_dim=64, nonlin=F.relu,
                norm_in=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(DNRI_Encoder, self).__init__()
        self.num_edges = 2
        self.dropout_prob = 0.3
        edges = np.ones(num_vars) - np.eye(num_vars)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)

        self.mlp1 = MLPNetwork(input_dim, hidden_dim,
                        hidden_dim=hidden_dim, constrain_out=False)
        self.mlp2 = MLPNetwork(hidden_dim * 2, hidden_dim,
                        hidden_dim=hidden_dim, constrain_out=False)
        self.mlp3 = MLPNetwork(hidden_dim, hidden_dim,
                        hidden_dim=hidden_dim, constrain_out=False)
        self.mlp4 = MLPNetwork(hidden_dim * 3, hidden_dim,
                        hidden_dim=hidden_dim, constrain_out=False)

        self.forward_rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.prior_fc_out = MLPNetwork(hidden_dim, self.num_edges,
                        hidden_dim=hidden_dim, constrain_out=False)

        self.num_vars = num_vars
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # Input size: [batch, num_vars, num_timesteps, embed_size]
        if len(node_embeddings.shape) == 4:
            send_embed = node_embeddings[:, self.send_edges, :, :]
            recv_embed = node_embeddings[:, self.recv_edges, :, :]
        else:
            send_embed = node_embeddings[:, self.send_edges, :]
            recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_vars-1) #TODO: do we want this average?

    def forward(self, inputs, prior_state):
        # Inputs is shape [batch, num_vars, input_size]
        x = self.mlp1(self.in_fn(inputs))  # 2-layer ELU net per node
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1)  # Skip connection
        x = self.mlp4(x)

        # At this point, x should be [batch, num_vars, hidden_size]
        # RNN aggregation
        old_shape = x.shape
        x  = x.contiguous().view(-1, 1, old_shape[-1])
        old_prior_shape = prior_state[0].shape
        if x.is_cuda:
            prior_state = (prior_state[0].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]).cuda(),
                            prior_state[1].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]).cuda())
        else:
            prior_state = (prior_state[0].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]),
                            prior_state[1].view(1, old_prior_shape[0]*old_prior_shape[1], old_prior_shape[2]))

        x, new_prior_state = self.forward_rnn(x, prior_state)
        prior_result = self.prior_fc_out(x).view(old_shape[0], old_shape[1], self.num_edges)
        new_prior_state = (new_prior_state[0].view(old_prior_shape), new_prior_state[1].view(old_prior_shape))
        return prior_result, new_prior_state