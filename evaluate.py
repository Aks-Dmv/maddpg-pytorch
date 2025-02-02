import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from gym.spaces import Box, Discrete

def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    if config.save_gifs:
        gif_path = model_path.parent / 'gifs'
        gif_path.mkdir(exist_ok=True)

    env = make_env(config.env_id, discrete_action=False)
    if isinstance(env.action_space[0], Box):
        discr_act = False
        get_shape = lambda x: x.shape[0]
    else:  # Discrete
        discr_act = True
        get_shape = lambda x: x.n
    num_out_pol = get_shape(env.action_space[0])
    
    agent_init_params = {'num_in_pol': env.observation_space[0].shape[0],
                        'num_out_pol': num_out_pol,
                        'num_vars': 3}
    maddpg = MADDPG(agent_init_params, 
                    nagents = 3,
                    hidden_dim=config.hidden_dim,
                    discrete_action=discr_act)
    save_dict = torch.load(model_path)
    maddpg.agents.load_params(save_dict['agent_params'])
    ifi = 1 / config.fps  # inter-frame interval

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        env.render('human')

        rnn_hidden = ( torch.zeros(1, config.n_rollout_threads * (maddpg.nagents)*(maddpg.nagents - 1), config.hidden_dim), 
                        torch.zeros(1, config.n_rollout_threads * (maddpg.nagents)*(maddpg.nagents - 1), config.hidden_dim) )
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions, new_rnn_hidden = maddpg.step(torch_obs, rnn_hidden, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions.cpu()]
            obs, rewards, dones, infos = env.step(actions)
            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            env.render('human')
            rnn_hidden = new_rnn_hidden
        if config.save_gifs:
            gif_num = 0
            while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
                gif_num += 1
            imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
                            frames, duration=ifi)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)

    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)

    config = parser.parse_args()

    run(config)