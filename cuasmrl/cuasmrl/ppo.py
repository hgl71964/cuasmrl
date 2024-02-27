import os
from dataclasses import asdict
from typing import Optional
import json


import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cuasmrl.utils.logger import get_logger

logger = get_logger(__name__)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):

    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.nvec = n_actions
        self.actor = layer_init(nn.Linear(128, self.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(
            hidden)


def env_loop(env, config):
    save_path = os.path.join(config.default_out_path, config.save_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.gpu == 1 else "cpu")
    logger.info(f"[ENV_LOOP] WorkDir: {save_path}; Device: {device}")

    # ===== log =====
    log = bool(config.log)
    if log:
        # t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # file = config.fn.split(
        #     "/")[-1] if config.fn is not None else config.dir
        # run_name = f"rlx_{config.env_id}__{config.agent}__{file}"
        # if config.annotation is not None:
        #     run_name += f"__{config.annotation}"
        # run_name += f"__{t}"
        # save_path = f"{config.default_out_path}/runs/{run_name}"
        writer = SummaryWriter(save_path)
        # https://github.com/abseil/abseil-py/issues/57

        config_dict = asdict(config)
        config_json = json.dumps(config_dict, indent=4)
        with open(os.path.join(save_path, "drl_config.json"), "w") as file:
            file.write(config_json)

        logger.info(f"[ENV_LOOP]save path: {save_path}")


    # ===== agent & opt =====
    agent = PPO(n_actions=env.action_space.nvec[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    # automatically load the latest ckpt
    ckpt_files = [f for f in os.listdir(save_path) if f.endswith('.pt')]
    latest_ckpt = None
    max_epoch = -1
    for file in ckpt_files:
        epoch_num = file.strip('.pt').split('_')[-1]
        logger.critical('xxxx', file, epoch_num)
        epoch_num = int(epoch_num)
        if epoch_num > max_epoch:
            max_epoch = epoch_num
            latest_ckpt = file

    if latest_ckpt is None:
        iteration = 0
    if latest_ckpt is not None:
        latest_ckpt_path = os.path.join(save_path, latest_ckpt)
        ckpt = torch.load(latest_ckpt_path)
        agent.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        iteration = ckpt['iteration'] + 1

    logger.info(f'[ENV_LOOP] start training from iteration {iteration}')

    # ===== constants =====
    anneal_lr = bool(config.anneal_lr)
    norm_adv = bool(config.norm_adv)
    clip_vloss = bool(config.clip_vloss)

    # ===== START GAME =====
    # ALGO Logic: Storage setup
    obs = torch.zeros((config.num_steps, config.num_env) +
                      env.observation_space.shape).to(device)
    actions = torch.zeros(
        (config.num_steps, config.num_env) + env.action_space.shape,
        dtype=torch.long).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_env)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_env)).to(device)
    dones = torch.zeros((config.num_steps, config.num_env)).to(device)
    values = torch.zeros((config.num_steps, config.num_env)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_env).to(device)

    for iteration in range(1, config.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.num_steps):
            global_step += config.num_env
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = env.step(
                action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        logger.info(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        if log:
                            writer.add_scalar("charts/episodic_return",
                                              info["episode"]["r"],
                                              global_step)
                            writer.add_scalar("charts/episodic_length",
                                              info["episode"]["l"],
                                              global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[
                    t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[
                    t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, ) + env.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, ) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs()
                                   > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),
                                         config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true -
                                                             y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if log:
            writer.add_scalar("charts/learning_rate",
                              optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(),
                              global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(),
                              global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(),
                              global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(),
                              global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs),
                              global_step)
            writer.add_scalar("losses/explained_variance", explained_var,
                              global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS",
                              int(global_step / (time.time() - start_time)),
                              global_step)

        if log:
            torch.save(
                {
                    'iteration': iteration,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"{save_path}/{config.agent}_ckpt_{iteration}.pt")

    # ===== STOP =====
    env.close()
    if log:
        writer.close()


def inference(env, config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and bool(config.gpu) else "cpu")
    logger.info(f"device: {device}")
    # ===== env =====
    state, _ = env.reset(seed=config.seed)

    # ===== agent =====
    assert config.weights_path is not None, "weights_path must be set"
    agent_id = config.agent_id if config.agent_id is not None else "agent-final"
    fn = os.path.join(f"{config.default_out_path}/runs/", config.weights_path,
                      f"{agent_id}.pt")
    agent = PPO()
    agent_state_dict = torch.load(fn, map_location=device)
    agent.load_state_dict(agent_state_dict)
    agent.to(device)

    next_obs = pyg.data.Batch.from_data_list([i[0] for i in state]).to(device)
    invalid_rule_mask = torch.cat([i[2] for i in state]).reshape(
        (env.num_env, -1)).to(device)

    # ==== rollouts ====
    cnt = 0
    t1 = time.perf_counter()
    inf_time = 0
    while True:
        cnt += 1
        with torch.no_grad():
            _t1 = time.perf_counter()
            action, _, _, _ = agent.get_action_and_value(
                next_obs,
                invalid_rule_mask=invalid_rule_mask,
            )
            _t2 = time.perf_counter()
        inf_time += _t2 - _t1

        # TRY NOT TO MODIFY: execute the game and log data.
        # print("a", action)
        # a = [tuple(i) for i in action.cpu()]
        next_obs, _, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        if done.all():
            break

        invalid_rule_mask = torch.cat([i[2] for i in next_obs]).reshape(
            (env.num_env, -1)).to(device)

        next_obs = pyg.data.Batch.from_data_list([i[0] for i in next_obs
                                                  ]).to(device)

        # logger.info(f"iter {cnt}; reward: {reward}")

    t2 = time.perf_counter()
    # print(terminated, truncated)
    return {
        "iter": cnt,
        "opt_time": t2 - t1,
        "inf_time": inf_time,
    }
