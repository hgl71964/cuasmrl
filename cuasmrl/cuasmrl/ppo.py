import os
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cuasmrl.utils.logger import get_logger
from cuasmrl.verify import test_via_cubin

logger = get_logger(__name__)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):

    def __init__(self, envs):
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
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n),
                                std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

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


def env_loop(envs, config):

    device = torch.device(
        "cuda" if torch.cuda.is_available() and bool(config.gpu) else "cpu")
    logger.info(f"device: {device}")
    # ===== env =====
    state, _ = envs.reset(seed=config.seed)

    # ===== log =====
    log = bool(config.l)
    if log:
        t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file = config.fn.split(
            "/")[-1] if config.fn is not None else config.dir
        run_name = f"rlx_{config.env_id}__{config.agent}__{file}"
        if config.annotation is not None:
            run_name += f"__{config.annotation}"
        run_name += f"__{t}"
        save_path = f"{config.default_out_path}/runs/{run_name}"
        writer = SummaryWriter(save_path)
        # https://github.com/abseil/abseil-py/issues/57
        config.append_flags_into_file(save_path + "/flags.txt")

        logger.info(f"[ENV_LOOP]save path: {save_path}")

    # ===== agent =====
    agent = PPO(
        actor_n_action=envs.single_action_space.nvec[0],
        n_node_features=envs.single_observation_space.node_space.shape[0],
        n_edge_features=envs.single_observation_space.edge_space.shape[0],
        num_head=config.num_head,
        n_layers=config.n_layers,
        hidden_size=config.hidden_size,
        vgat=config.vgat,
        use_dropout=bool(config.use_dropout),
        device=device,
    ).to(device)

    if config.weights_path is not None:
        # TODO save+load opt states too?
        logger.warning(
            f"[ENV_LOOP] fine tuning from checkpoint {config.weights_path}")
        agent_id = config.agent_id if config.agent_id is not None else "agent-final"
        fn = os.path.join(f"{config.default_out_path}/runs/",
                          config.weights_path, f"{agent_id}.pt")
        agent_state_dict = torch.load(fn, map_location=device)
        agent.load_state_dict(agent_state_dict)
        agent.fine_tuning()
        agent.to(device)

    optimizer = optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    # ===== START GAME =====
    # ALGO Logic: Storage setup
    # gh512
    # obs = torch.zeros((config.num_steps, config.num_envs) +
    #                   envs.single_observation_space.shape).to(device)
    # obs = []  # collect graphs
    actions = torch.zeros(
        # (config.num_steps, config.num_envs) + envs.single_action_space.shape,
        (config.num_steps, config.num_envs),
        dtype=torch.long).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    invalid_rule_masks = torch.zeros(
        (config.num_steps, config.num_envs, envs.single_action_space.nvec[0]),
        dtype=torch.long).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # gh512
    # next_obs = torch.Tensor(envs.reset()).to(device)
    next_obs = pyg.data.Batch.from_data_list([i[0] for i in state]).to(device)
    next_done = torch.zeros(config.num_envs).to(device)
    invalid_rule_mask = torch.cat([i[2] for i in state]).reshape(
        (config.num_envs, -1)).to(device)

    # batch size
    batch_size = int(config.num_envs * config.num_steps)
    minibatch_size = int(batch_size // config.num_mini_batch)
    num_updates = config.total_timesteps // batch_size
    logger.info(f"[ENV_LOOP] minibatch size: {minibatch_size}")

    # resolve int config
    anneal_lr = bool(config.anneal_lr)
    gae = bool(config.gae)
    norm_adv = bool(config.norm_adv)
    clip_vloss = bool(config.clip_vloss)

    # utility
    save_freq = (num_updates + 1) // 10
    best_episodic_return = -float("inf")

    # ===== RUN =====
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.lr
            optimizer.param_groups[0]["lr"] = lrnow

        # ==== rollouts ====
        # gh512, reset;
        obs = []  # list[pyg.Data]; reset each update
        for step in range(0, config.num_steps):
            global_step += 1 * config.num_envs
            # gh512
            # obs[step] = next_obs
            obs.append(next_obs)
            dones[step] = next_done

            # print(invalid_rule_masks.shape)
            # print(invalid_rule_masks[step].shape)
            # print(invalid_rule_mask.shape)
            invalid_rule_masks[step] = invalid_rule_mask

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs,
                    invalid_rule_mask=invalid_rule_mask,
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # gh512
            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # gh512
            # next_obs, next_done = torch.Tensor(next_obs).to(
            #     device), torch.Tensor(done).to(device)
            next_done = torch.Tensor(done).to(device)
            invalid_rule_mask = torch.cat([i[2] for i in next_obs]).reshape(
                (config.num_envs, -1)).to(device)
            next_obs = pyg.data.Batch.from_data_list([i[0] for i in next_obs
                                                      ]).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                # this is recorded by RecordEpisodeStatistics wrapper
                logger.info(
                    f"[ENV_LOOP] global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}, episodic_time={info['episode']['t']}"
                )
                if log:
                    writer.add_scalar("charts/episodic_return",
                                      info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length",
                                      info["episode"]["l"], global_step)

        # ==== after rollouts, updates ====
        # bootstrap value if not done
        # print(f"[gae] {update}", end="\t")
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if gae:
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
                        t] + config.gamma * nextvalues * nextnonterminal - values[
                            t]
                    advantages[
                        t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[
                        t] + config.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        # gh512
        # b_obs = obs.reshape((-1, ) + envs.single_observation_space.shape)
        # NOTE: `unbatch` graphs
        b_obs = []
        for batch_graph in obs:
            per_step_graphs = pyg.data.Batch.to_data_list(batch_graph)
            for per_step_per_env_graph in per_step_graphs:
                b_obs.append(per_step_per_env_graph)

        # those buffers are in device
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_invalid_rule_masks = invalid_rule_masks.reshape(
            config.num_steps * config.num_envs, -1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        # print(f"[update] {update}")
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # print("indx")
                # print(batch_size)
                # print(mb_inds)
                # print(len(b_obs))
                # print(b_obs[0])
                # tmp = pyg.data.Batch.to_data_list(b_obs[0])
                # print("tmp")
                # print(len(tmp))

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    # gh512 (re-batch here)
                    # b_obs[mb_inds],
                    pyg.data.Batch.from_data_list([b_obs[i] for i in mb_inds]
                                                  ).to(device),
                    invalid_rule_mask=b_invalid_rule_masks[mb_inds],
                    action=b_actions.long()[mb_inds],
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

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        # ==== after updates, log ====
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true -
                                                             y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        t = int(global_step / (time.time() - start_time))
        logger.info(f"[ENV_LOOP] Update: {update}/{num_updates} SPS: {t}")
        # logger.info(f"[ENV_LOOP] episodic_return: {episodic_return}")
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
            writer.add_scalar("charts/SPS", t, global_step)

        if log and update % save_freq == 0:
            torch.save(agent.state_dict(),
                       f"{save_path}/agent-{global_step}.pt")

    # ===== STOP =====
    envs.close()
    if log:
        # save
        torch.save(agent.state_dict(), f"{save_path}/agent-final.pt")
        writer.close()


def inference(envs, config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and bool(config.gpu) else "cpu")
    logger.info(f"device: {device}")
    # ===== envs =====
    state, _ = envs.reset(seed=config.seed)

    # ===== agent =====
    assert config.weights_path is not None, "weights_path must be set"
    agent_id = config.agent_id if config.agent_id is not None else "agent-final"
    fn = os.path.join(f"{config.default_out_path}/runs/", config.weights_path,
                      f"{agent_id}.pt")
    agent = PPO(
        actor_n_action=envs.single_action_space.nvec[0],
        n_node_features=envs.single_observation_space.node_space.shape[0],
        n_edge_features=envs.single_observation_space.edge_space.shape[0],
        num_head=config.num_head,
        n_layers=config.n_layers,
        hidden_size=config.hidden_size,
        vgat=config.vgat,
        use_dropout=bool(config.use_dropout),
        device=device,
    )
    agent_state_dict = torch.load(fn, map_location=device)
    agent.load_state_dict(agent_state_dict)
    agent.to(device)

    next_obs = pyg.data.Batch.from_data_list([i[0] for i in state]).to(device)
    invalid_rule_mask = torch.cat([i[2] for i in state]).reshape(
        (envs.num_envs, -1)).to(device)

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
        next_obs, _, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        if done.all():
            break

        invalid_rule_mask = torch.cat([i[2] for i in next_obs]).reshape(
            (envs.num_envs, -1)).to(device)

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
