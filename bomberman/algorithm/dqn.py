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


# 2.1.1. POE
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replayを実現するためのメモリクラス.
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.priorities[0] = 1.0

    def __len__(self):
        return len(self.buffer)

    # 経験をリプレイバッファに保存する． 経験は(obs, action, reward, next_obs, done)の5つ組を想定
    def push(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience

        # 優先度は最初は大きな値で初期化しておき, 後でサンプルされた時に更新する
        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        # 現在経験が入っている部分に対応する優先度を取り出し, サンプルする確率を計算
        priorities = self.priorities[
            : self.buffer_size if len(self.buffer) == self.buffer_size else self.index
        ]
        priorities = priorities**alpha
        # 確率の総和が0になるよう調整
        prob = priorities / priorities.sum()

        # >> 演習: 確率probに従ってサンプルする経験のインデックスを用意しましょう
        # ヒント: np.random.choice などが便利です
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)

        # >> 演習: 上式の通りに重点サンプリングの補正のための重みを計算してみましょう
        weights = (len(self.buffer) * prob[indices]) ** (-beta)
        weights = weights / weights.max()

        # 上でサンプルしたインデックスに基づいて経験をサンプルし, (obs, action, reward, next_obs, done)に分ける
        obs, action, reward, next_obs, done = zip(*[self.buffer[i] for i in indices])

        # あとで計算しやすいようにtorch.Tensorに変換して(obs, action, reward, next_obs, done, indices, weights)の7つ組を返す
        return (
            torch.stack(obs),
            torch.as_tensor(action),
            torch.as_tensor(reward, dtype=torch.float32),
            torch.stack(next_obs),
            torch.as_tensor(done, dtype=torch.uint8),
            indices,
            torch.as_tensor(weights, dtype=torch.float32),
        )

    # 優先度を更新する. 優先度が極端に小さくなって経験が全く選ばれないということがないように, 微小値を加算しておく.
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-4


# 2.1.2 Dueling Network
class CNNQNetwork(nn.Module):
    """
    Dueling Networkを用いたQ関数を実現するためのニューラルネットワークをクラスとして記述します.
    """

    def __init__(self, state_shape, n_action):
        super(CNNQNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_action = n_action
        # Dueling Networkでも, 畳込み部分は共有する
        # ToDo conv2dの計算
        # https://ichi.pro/conv-2-d-saigo-ni-fuxowa-do-pasu-de-nani-ga-okoru-ka-o-rikaisuru-30488625459528
        # 一つの辺は(元チャンネル - kernel_size + 1) になる
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=state_shape[0], out_channels=32, kernel_size=3, stride=1
            ),  # 1 x windowX x windowY
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=1),  # 32x11x9 -> 48x9x7
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),  # 48x9x7 -> 64x7x5
            nn.ReLU(),
        )

        # Dueling Networkのための分岐した全結合層
        # 状態価値
        self.fc_state = nn.Sequential(
            # qqhann: なぜ-6なのかよくわかってない
            nn.Linear(64 * (self.state_shape[1] - 6) * (self.state_shape[2] - 6), 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # アドバンテージ
        self.fc_advantage = nn.Sequential(
            # qqhann: なぜ-6なのかよくわかってない
            nn.Linear(64 * (self.state_shape[1] - 6) * (self.state_shape[2] - 6), 512),
            nn.ReLU(),
            nn.Linear(512, n_action),
        )

    def forward(self, obs):
        feature = self.conv_layers(obs)
        feature = feature.view(feature.size(0), -1)  # Flatten. 64x7x7　-> 3136

        state_values = self.fc_state(feature)
        advantage = self.fc_advantage(feature)

        # 状態価値 + アドバンテージ で行動価値を計算しますが、安定化のためアドバンテージの（行動間での）平均を引きます
        action_values = (
            state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
        )
        return action_values

    # epsilon-greedy. 確率epsilonでランダムに行動し, それ以外はニューラルネットワークの予測結果に基づいてgreedyに行動します.
    def act(self, obs, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.n_action)
        else:
            # 行動を選択する時には勾配を追跡する必要がない
            with torch.no_grad():
                action = torch.argmax(self.forward(obs.unsqueeze(0))).item()
        return action
