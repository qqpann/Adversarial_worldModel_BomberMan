import copy
import pickle
import pprint
import random

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torchvision.transforms as transforms
from PIL import Image
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import bomberman
from bomberman.algorithm.dqn import CNNQNetwork, PrioritizedReplayBuffer
from bomberman.common.utils import init_device
from bomberman.common.variables import *

path = "."


def saveModels(nets, target_nets, optimizers):
    with open(path + "/weights.pickle", "wb") as f:
        pickle.dump([nets, target_nets, optimizers], f)  # 2.1.3 ハイパラ


def main():
    env = bomberman.Env()
    device = init_device()

    """
    リプレイバッファの宣言
    """
    buffer_size = 10000  # 　リプレイバッファに入る経験の最大数
    initial_buffer_size = 1000  # 学習を開始する最低限の経験の数
    replay_buffers = []
    for _ in range(player_count):
        replay_buffers.append(PrioritizedReplayBuffer(buffer_size))

    """
        ネットワークの宣言
    """
    nets = []
    target_nets = []
    for _ in range(player_count):
        nets.append(
            CNNQNetwork(env.observation_space.shape, n_action=env.action_space.n).to(
                device
            )
        )
        target_nets.append(
            CNNQNetwork(env.observation_space.shape, n_action=env.action_space.n).to(
                device
            )
        )
    target_update_interval = 2000  # 学習安定化のために用いるターゲットネットワークの同期間隔

    """
        オプティマイザとロス関数の宣言
    """
    optimizers = []
    for net in nets:
        optimizers.append(optim.Adam(net.parameters(), lr=1e-4))  # オプティマイザはAdam
    loss_func = nn.SmoothL1Loss(reduction="none")  # ロスはSmoothL1loss（別名Huber loss）

    """
        Prioritized Experience Replayのためのパラメータβ
    """
    beta_begin = 0.4
    beta_end = 1.0
    beta_decay = 500000
    # beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす
    beta_func = lambda step: min(
        beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay)
    )

    """
        探索のためのパラメータε
    """
    epsilon_begin = 1.0
    epsilon_end = 0.01
    epsilon_decay = 50000
    # epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
    epsilon_func = lambda step: max(
        epsilon_end,
        epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay),
    )

    """
        その他のハイパーパラメータ
    """
    gamma = 0.99  # 　割引率
    batch_size = 64
    # n_episodes = 3000  # 学習を行うエピソード数
    n_episodes = 6000  # 動作確認用

    # 2.1.4 Double DQN
    def update(batch_size, beta, pid):
        obs, action, reward, next_obs, done, indices, weights = replay_buffers[
            pid
        ].sample(batch_size, beta)
        obs, action, reward, next_obs, done, weights = (
            obs.float().to(device),
            action.to(device),
            reward.to(device),
            next_obs.float().to(device),
            done.to(device),
            weights.to(device),
        )

        # ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
        q_values = nets[pid](obs).gather(1, action.unsqueeze(1)).squeeze(1)

        # 目標値の計算なので勾配を追跡しない
        with torch.no_grad():
            # Double DQN.
            # >> 演習: Double DQNのターゲット価値の計算を実装してみましょう
            # ① 現在のQ関数でgreedyに行動を選択し,
            greedy_action_next = torch.argmax(nets[pid](next_obs), dim=1)
            # ②　対応する価値はターゲットネットワークのものを参照します.
            q_values_next = (
                target_nets[pid](next_obs)
                .gather(1, greedy_action_next.unsqueeze(1))
                .squeeze(1)
            )

        # ベルマン方程式に基づき, 更新先の価値を計算します.
        # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
        target_q_values = reward + gamma * q_values_next * (1 - done)

        # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
        optimizers[pid].zero_grad()
        loss = (weights * loss_func(q_values, target_q_values)).mean()
        loss.backward()
        optimizers[pid].step()

        # TD誤差に基づいて, サンプルされた経験の優先度を更新します.
        replay_buffers[pid].update_priorities(
            indices, (target_q_values - q_values).abs().detach().cpu().numpy()
        )

        return loss.item()

    writer = None

    step = 0
    best_env = copy.deepcopy(env)

    for episode in range(n_episodes):
        cnt = 0
        obss = env.reset()
        done = False
        total_rewards = [0 for i in range(player_count)]

        while not done:
            actions = [0 for i in range(player_count)]
            for pid in range(player_count):
                # ε-greedyで行動を選択
                actions[pid] = nets[pid].act(
                    obss[pid].float().to(device), epsilon_func(step)
                )
                if pid == 0:
                    env.addAction(pid, actions[pid])
                else:
                    env.addAction(pid, 0)  # 動かない

            # 環境中で実際に行動
            next_obss, rewards, done, _ = env.step()

            # プレイヤー0のみを学習
            for pid in range(1):
                total_rewards[pid] += rewards[pid]
                # リプレイバッファに経験を蓄積
                replay_buffers[pid].push(
                    [obss[pid], actions[pid], rewards[pid], next_obss[pid], done]
                )
                obss[pid] = next_obss[pid]

                # ネットワークを更新
                if len(replay_buffers[pid]) > initial_buffer_size:
                    update(batch_size, beta_func(step), pid)

                # ターゲットネットワークを定期的に同期させる
                if (step + 1) % target_update_interval == 0:
                    target_nets[pid].load_state_dict(nets[pid].state_dict())

            step += 1
            cnt += 1
            # 遅延行為を覚えた場合の対策
            if cnt >= 300:
                print("MAX ACTION")
                break

        if episode % 200 == 0:
            env.render_to_gif(0, path + "/result/Episode" + str(episode))
            # 200エピソードごとに相手の重みを同期
            nets[1] = copy.deepcopy(nets[0])

        # 1000エピソードごとに重み保存
        if episode % 1000 == 0:
            saveModels(nets, target_nets, optimizers)

        print(
            "Episode: {},  Step: {},  Reward: {}".format(
                episode + 1, step + 1, total_rewards[0]
            )
        )
        if writer:
            writer.add_scalar("Reward", total_rewards[0], episode)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
