import os
import sys
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
from cuasmrl.utils.constants import Status
from cuasmrl.utils.record import save_data

logger = get_logger(__name__)


class CategoricalMasked(Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.device = torch.device("cpu")  # XXX hardcore for now
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits,
                                                    validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(self.masks, logits,
                                 torch.tensor(-1e8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits,
                                                    validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p,
                              torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):

    def __init__(self, observation_space, action_space):
        super().__init__()
        *_, H, W = observation_space.shape

        # after = (before + padding*2 - kernel_size//2) // stride
        H1, W1 = (H + 2 - 4) // 2, W
        H2, W2 = (H1 + 0 - 4 // 2) // 2, (W1 + 1 * 2 - 4 // 2) // 2
        H3, W3 = H2 - 2, W2 - 2
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, (5, 3), stride=(2, 1), padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=(0, 1))),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            # nn.Flatten(0, -1),
            nn.Flatten(),  # [batch_size, whatever]
            layer_init(nn.Linear(64 * H3 * W3, 512)),
            nn.ReLU(),
        )
        # multiDiscrete
        # self.nvec = nvec
        # self.actor = layer_init(nn.Linear(512, self.nvec.sum()), std=0.01)
        # self.critic = layer_init(nn.Linear(512, 1), std=1)

        # Discrete (flattened)
        self.actor = layer_init(nn.Linear(512, action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action_masks, action=None):
        if len(x.shape) == 3:
            # add a batch dim when inference
            x = x.unsqueeze(0)
        hidden = self.network(x)

        logits = self.actor(hidden)  # [batch_size, n]

        categorical = CategoricalMasked(logits=logits, masks=action_masks)

        if action is None:
            action = categorical.sample()
        logprob = categorical.log_prob(action)
        entropy = categorical.entropy()

        # XXX: mask has inter=dependencies; might as well just flatten action space...
        # split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        # split_action_masks = torch.split(action_masks,
        #                                  self.nvec.tolist(),
        #                                  dim=1)
        # multi_categoricals = [
        #     CategoricalMasked(logits=logits, masks=action_masks)
        #     for (logits, iam) in zip([logits], split_action_masks)
        # ]
        # if action is None:
        # action = torch.stack(
        #     [categorical.sample() for categorical in multi_categoricals])
        # logprob = torch.stack([
        #     categorical.log_prob(a)
        #     for a, categorical in zip(action, multi_categoricals)
        # ])
        # entropy = torch.stack(
        #     [categorical.entropy() for categorical in multi_categoricals])
        # return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden)
        return action, logprob, entropy, self.critic(hidden)


def env_loop(env, config):
    save_path = os.path.join(config.default_out_path, config.save_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.gpu == 1 else "cpu")
    logger.info(f"[ENV_LOOP] WorkDir: {save_path}; Traing Device: {device}")

    profile = False
    if os.getenv("SIP_PROFILE", "0") == "1":
        profile = True

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
        # https://github.com/abseil/abseil-py/issues/57

        # config
        config_dict = asdict(config)
        config_json = json.dumps(config_dict, indent=4)

        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "drl_config.json"), "w") as file:
            file.write(config_json)

        # tensorboard (TensorBoard can visualize multiple files in the same log directory)
        writer = SummaryWriter(save_path)

    # ===== agent & opt =====
    agent = PPO(env.observation_space, env.action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    # load the latest ckpt and clean up
    ckpt_files = [f for f in os.listdir(save_path) if f.endswith('.pt')]
    sorted_ckpt_files = sorted(
        ckpt_files,
        key=lambda x: int(x.strip('.pt').split('_')[-1]),
        reverse=True)
    latest_5_ckpts = sorted_ckpt_files[:5]
    for ckpt_file in ckpt_files:
        if ckpt_file not in latest_5_ckpts:
            os.remove(os.path.join(save_path, ckpt_file))
            logger.warning(f"Deleted {ckpt_file}")

    latest_ckpt = None
    max_iteration = -1
    for file in latest_5_ckpts:
        iteration_num = int(file.strip('.pt').split('_')[-1])
        if iteration_num > max_iteration:
            max_iteration = iteration_num
            latest_ckpt = file

    if latest_ckpt is None:
        start_iteration = 1
        global_step = 0
        best_reward = 0
    else:
        latest_ckpt_path = os.path.join(save_path, latest_ckpt)
        ckpt = torch.load(latest_ckpt_path)
        agent.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iteration = ckpt['iteration'] + 1
        global_step = ckpt['global_step']
        best_reward = ckpt['best_reward']

    if start_iteration >= config.num_iterations:
        info = {'status': Status.OK}  # will skip the loop

    logger.info(f'[ENV_LOOP] start training from iteration {start_iteration}')

    # ===== constants =====
    batch_size = config.num_env * config.num_steps
    anneal_lr = bool(config.anneal_lr)
    norm_adv = bool(config.norm_adv)
    clip_vloss = bool(config.clip_vloss)

    # ===== START GAME =====
    # ALGO Logic: Storage setup
    # obs = torch.zeros((config.num_steps, config.num_env) +
    #                   env.observation_space.shape).to(device)
    obs = torch.zeros((config.num_steps, ) +
                      env.observation_space.shape).to(device)
    actions = torch.zeros(
        (config.num_steps, config.num_env) + env.action_space.shape,
        dtype=torch.long).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_env)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_env)).to(device)
    dones = torch.zeros((config.num_steps, config.num_env)).to(device)
    values = torch.zeros((config.num_steps, config.num_env)).to(device)
    action_masks = torch.zeros(
        (config.num_steps, config.num_env, env.action_space.n),
        dtype=torch.int32).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, info = env.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_env).to(device)

    for iteration in range(start_iteration, config.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.lr
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.num_steps):
            global_step += config.num_env
            obs[step] = next_obs
            dones[step] = next_done
            # append 1 to indicate noop
            mask = torch.tensor(info['masks'], dtype=torch.int32).flatten()
            # mask = torch.cat([mask, torch.tensor([1], dtype=torch.int32)])
            mask = torch.cat([mask, torch.tensor([0], dtype=torch.int32)])
            action_masks[step] = mask
            if profile:
                logger.info(f'effective dim: {action_masks[step].sum()}')

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs,
                    action_masks[step],
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, info = env.step(
                action.cpu().numpy())
            next_done_np: bool = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor([next_done_np]).to(device)

            # print(action)
            # print(reward)

            # handle error
            if info['status'] == Status.SEGFAULT:
                # subsequent call will be error;
                # save and relaunch
                # TODO break will stitll train; should we exit here?

                # NOTE: the Adam.step() will check CUDA stream,
                # and causes an error when SEGFAULT
                # we want to train in CPU since all tensors are in CPU
                # so bypass the test
                # hopefully this leads to training and then exit this process
                torch.backends.cuda.is_built = lambda: False
                break
            elif info['status'] == Status.TESTFAIL or next_done_np:
                # before reset save the best cubin
                if info['status'] is not Status.TESTFAIL and global_step > 1000:
                    if 'episode' in info and env.unwrapped.last_perf > best_reward:
                        best_reward = env.unwrapped.last_perf
                        # assemble and save
                        env.unwrapped.eng.assemble(env.unwrapped.sample)
                        p = save_data(
                            env.unwrapped.eng.bin,
                            env.unwrapped.last_perf,
                            env.unwrapped.init_perf,
                            save_path,
                        )
                        logger.info(
                            f'save cubin with {best_reward} at {iteration} to {p}'
                        )

                # log
                if 'episode' in info:
                    logger.info(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}, best_reward {best_reward:.2f}"
                    )
                    if log:
                        writer.add_scalar("charts/episodic_return",
                                          info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length",
                                          info["episode"]["l"], global_step)
                        writer.add_scalar("charts/init_perf",
                                          env.unwrapped.init_perf, global_step)

                # in SyncVectorEnv, the env is automatically reset if done
                # we try to do the same here
                next_obs, info = env.reset(seed=config.seed)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(config.num_env).to(device)

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
        b_actions = actions.reshape((-1, ) + env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_action_masks[mb_inds],
                    b_actions[mb_inds],
                )
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
            print(f"SPS: {global_step / (time.time() - start_time):.2f}")
            writer.add_scalar("charts/SPS",
                              int(global_step / (time.time() - start_time)),
                              global_step)

        if log:
            if info['status'] == Status.SEGFAULT:
                torch.save(
                    {
                        'iteration': iteration,
                        'global_step': global_step,
                        'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_reward': best_reward,
                    }, f"{save_path}/{config.agent}_ckpt_{iteration}.pt")
                logger.warning(
                    f'save {config.agent} at {iteration} for segfault')
            elif iteration % config.ckpt_freq == 0:
                torch.save(
                    {
                        'iteration': iteration,
                        'global_step': global_step,
                        'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_reward': best_reward,
                    }, f"{save_path}/{config.agent}_ckpt_{iteration}.pt")
                logger.info(f'save {config.agent} at {iteration}')

                # clean-up
                time.sleep(1)
                ckpt_files = [
                    f for f in os.listdir(save_path) if f.endswith('.pt')
                ]
                sorted_ckpt_files = sorted(
                    ckpt_files,
                    key=lambda x: int(x.strip('.pt').split('_')[-1]),
                    reverse=True)
                latest_5_ckpts = sorted_ckpt_files[:5]
                for ckpt_file in ckpt_files:
                    if ckpt_file not in latest_5_ckpts:
                        os.remove(os.path.join(save_path, ckpt_file))
                        logger.warning(f"Deleted {ckpt_file}")

        # we have to exit
        if info['status'] == Status.SEGFAULT:
            break

    # ===== STOP =====
    env.close()
    if log:
        writer.close()

    if info['status'] == Status.SEGFAULT:
        sys.exit(1)  # inform the driver to re-launch


def inference(env, config):
    save_path = os.path.join(config.default_out_path, config.save_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.gpu == 1 else "cpu")
    logger.info(f"[INFERENCE] WorkDir: {save_path}; Device: {device}")

    profile = False
    if os.getenv("SIP_PROFILE", "0") == "1":
        profile = True

    # ===== agent & opt =====
    agent = PPO(env.observation_space, env.action_space).to(device)

    # load the latest ckpt
    ckpt_files = [f for f in os.listdir(save_path) if f.endswith('.pt')]
    sorted_ckpt_files = sorted(
        ckpt_files,
        key=lambda x: int(x.strip('.pt').split('_')[-1]),
        reverse=True)
    latest_5_ckpts = sorted_ckpt_files[:5]

    latest_ckpt = None
    max_iteration = -1
    for file in latest_5_ckpts:
        iteration_num = int(file.strip('.pt').split('_')[-1])
        if iteration_num > max_iteration:
            max_iteration = iteration_num
            latest_ckpt = file

    if latest_ckpt is None:
        raise RuntimeError(f'cannot find ckpt at {save_path}')
    else:
        latest_ckpt_path = os.path.join(save_path, latest_ckpt)
        ckpt = torch.load(latest_ckpt_path)
        agent.load_state_dict(ckpt['model_state_dict'])
        start_iteration = ckpt['iteration'] + 1
        global_step = ckpt['global_step']
        best_reward = ckpt['best_reward']

    logger.info(f'[INFERENCE] from iteration {start_iteration}')

    # TRY NOT TO MODIFY: start the game
    next_obs, info = env.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_env).to(device)

    for step in range(0, config.num_steps):
        # append 1 to indicate noop
        mask = torch.tensor(info['masks'], dtype=torch.int32).flatten()
        # mask = torch.cat([mask, torch.tensor([1], dtype=torch.int32)])
        mask = torch.cat([mask, torch.tensor([0], dtype=torch.int32)])
        if profile:
            logger.info(f'effective dim: {mask.sum()}')

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(
                next_obs,
                mask,
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        np_action = action.cpu().numpy()
        next_obs, reward, terminations, truncations, info = env.step(np_action)

        next_done_np: bool = np.logical_or(terminations, truncations)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.Tensor([next_done_np]).to(device)

        # handle error
        if info['status'] == Status.SEGFAULT:
            # subsequent call will be error;
            raise RuntimeError('seg fault')
        elif info['status'] == Status.TESTFAIL or next_done_np:
            # log
            if 'episode' in info:
                logger.info(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}, best_reward {best_reward:.2f}"
                )

            # in SyncVectorEnv, the env is automatically reset if done
            # we try to do the same here
            # next_obs, info = env.reset(seed=config.seed)
            # next_obs = torch.Tensor(next_obs).to(device)
            # next_done = torch.zeros(config.num_env).to(device)
            break

    # ===== STOP =====
    env.close()
