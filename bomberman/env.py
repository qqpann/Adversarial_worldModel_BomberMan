import copy
import pickle
import pprint
import random

import cv2
import gym

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils

# import torchvision.transforms as transforms
from PIL import Image
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from bomberman.common.utils import isInBoard, pos2v, rand_ints_nodup
from bomberman.common.variables import (
    ALLIVE,
    CLOSEBONUS,
    INVALID_ACTION_PENALTY,
    ITEMS,
    LOSE,
    MAXBOMBS,
    ONBOMHIT,
    ONBOMHITTED,
    ONBOMSELF,
    ONBREAKBRICK,
    WIN,
    V,
    boardXsize,
    boardYsize,
    brickCount,
    n_actions,
    player_count,
    windowXsize,
    windowYsize,
)
from bomberman.obj import Bomb, Player

# 人間の目に優しいように
path = "../.."
chip_none = cv2.imread(path + "/assets/none.jpg")
chip_block = cv2.imread(path + "/assets/block.jpg")
chip_bomb = cv2.imread(path + "/assets/bom.jpg")
chip_brick = cv2.imread(path + "/assets/brick.jpg")
chip_enemy = cv2.imread(path + "/assets/enemy.jpg")
chip_player = cv2.imread(path + "/assets/player.jpg")
chip_bomb_on_me = cv2.imread(path + "/assets/bomonme.jpg")
chip_bomb_on_enemy = cv2.imread(path + "/assets/bomonenemy.jpg")


class Env(gym.Env):
    def __init__(self, logDisp=False):
        self.reset()
        # 以下gym用
        self.action_space = gym.spaces.Discrete(6)
        LOW = np.array([0 for i in range(boardXsize * boardYsize)]).reshape(
            1, boardXsize, boardYsize
        )
        HIGH = np.array([1 for i in range(boardXsize * boardYsize)]).reshape(
            1, boardXsize, boardYsize
        )
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH)
        self.logDisp = logDisp

    def reset(self):
        self.isFin = False
        self.actions = [0 for i in range(player_count)]
        self.playableId = [1 for i in range(player_count)]
        self.players = [Player(i) for i in range(player_count)]
        self.bombs = [Bomb() for i in range(MAXBOMBS)]
        self.boardHistory = []
        self.resetBoard()
        self.boardHistory.append(copy.deepcopy(self.board))
        return self.observations()

    # 盤面リセット
    def resetBoard(self):
        # 盤面初期化
        self.board = np.zeros([boardXsize, boardYsize])

        # 破壊不可オブジェクト配置
        for x in range(1, boardXsize, 2):
            for y in range(1, boardYsize, 2):
                self.board[x, y] = 1

        # プレイヤー初期位置定義
        init_pos = np.array(
            [
                [0, 0],
                [boardXsize - 1, boardYsize - 1],
                [0, boardYsize - 1],
                [boardXsize - 1, 0],
            ]
        )
        for i in range(player_count):
            self.board[pos2v(init_pos[i])] = 10 + i
            self.players[i]._changePos(pos=init_pos[i])

        self.setBrick(brickCount)

    # レンガ配置
    # プレイヤーの上下左右1マスは設置不可
    def setBrick(self, cnt):
        unsettable = []
        for player in self.players:
            for v in V:
                p = np.add(player.pos, v)
                if isInBoard(p) and self.board[pos2v(p)] == 0:
                    unsettable.append(p)

        settable = []
        for x, board_x in enumerate(self.board):
            for y, board_elm in enumerate(board_x):
                if board_elm == 0 and not any(
                    [np.allclose(p, [x, y]) for p in unsettable]
                ):
                    settable.append([x, y])

        bricks_pos = rand_ints_nodup(0, len(settable), cnt)
        for b in bricks_pos:
            self.board[pos2v(settable[b])] = 2

    # キャラの移動処理
    def move(self, pid, direction):
        p = np.add(self.players[pid].pos, V[direction])
        if self.logDisp:
            print(f"{pid}は{p}に移動")
        if isInBoard(p):
            # 空白の時のみ移動可能
            if self.board[pos2v(p)] == 0:
                self.board[pos2v(self.players[pid].pos)] -= pid + 10
                self.board[pos2v(p)] += pid + 10
                self.players[pid]._changePos(p)
            else:
                self.onCollision(pid)
        else:
            self.onCollision(pid)

    # 壁など衝突 ToDo:減点処理...?
    def onCollision(self, pid):
        if self.logDisp:
            print(f"{pid}:ごつん")
        self.players[pid].scoring(INVALID_ACTION_PENALTY)

    # 爆弾セット
    def bombSet(self, pid, pos):
        # 爆弾がすでにあるかおける上限を超えたとき処理をスキップ
        if (
            self.board[pos2v(pos)] >= 100
            or self.players[pid].setBomb >= self.players[pid].bombMax
        ):
            # ToDo:減点処理...?
            if self.logDisp:
                print(f"pid:{pid}は爆弾が置けない")
            self.players[pid].scoring(INVALID_ACTION_PENALTY)
            return
        for i in range(MAXBOMBS):
            if not self.bombs[i].isActive:
                self.bombs[i].activate(pos, pid, self.players[pid].bombLen)
                self.board[pos2v(pos)] += 100
                self.players[pid].setBomb += 1
                break

    # 爆弾tick
    def bombTick(self):
        for i in range(MAXBOMBS):
            if self.bombs[i].isActive:
                self.bombs[i].timer -= 1
                if self.bombs[i].timer <= 0:
                    self.bombDmg(i)
                    self.bombs[i].__init__()

    # 爆弾爆発処理
    def bombDmg(self, bomId):
        pos = self.bombs[bomId].pos
        if self.logDisp:
            print(f"{pos}で{self.bombs[bomId].who}の爆弾が長さ{self.bombs[bomId].len}で爆発")
        # 爆弾を置いた人が置けるようにする
        self.players[self.bombs[bomId].who].setBomb -= 1
        # 爆発跡地
        self.board[pos2v(pos)] -= 100
        self.exploseCheck(pos, bomId)
        for v in V:
            for i in range(1, self.bombs[bomId].len):
                p = np.add(pos, v * i)
                if isInBoard(p):
                    if self.exploseCheck(p, bomId) == 1:
                        break
                else:
                    break

    # 爆発後の地形変化 1を返すとき爆発がそこで止まることを示している
    def exploseCheck(self, p, bomId):
        whosBom = self.bombs[bomId].who
        if self.logDisp:
            print(f"{p}は{self.board[pos2v(p)]}")
        # 破壊不能の場合終了
        if int(self.board[pos2v(p)]) == 1:
            return 1
        # レンガの場合レンガを破壊して終了
        elif int(self.board[pos2v(p)]) == 2:
            self.board[pos2v(p)] = 0
            self.players[whosBom].scoring(ONBREAKBRICK)
            return 1
        # 爆弾の場合次のtickに連鎖爆発
        elif int(self.board[pos2v(p)]) >= 100:
            for i in range(MAXBOMBS):
                if np.allclose(self.bombs[i].pos, p):
                    if self.bombs[i].timer >= 2:
                        self.bombs[i].timer = 1
                    return 0
        # プレイヤーの場合ダメージ処理
        elif int(self.board[pos2v(p)]) >= 10:
            damagedPid = int(self.board[pos2v(p)]) % 10
            self.players[damagedPid].dmg()

            if self.logDisp:
                print(f"{damagedPid}が被弾")

            # 爆弾を当てたrewardと被弾reward
            if whosBom != damagedPid:
                self.players[whosBom].scoring(ONBOMHIT)
                self.players[damagedPid].scoring(ONBOMHITTED)
            else:
                self.players[damagedPid].scoring(ONBOMSELF)

            if not self.players[damagedPid].isSurvive():
                self.playableId[damagedPid] = 0
                self.players[damagedPid].scoring(LOSE)
                if whosBom != damagedPid:
                    self.players[whosBom].scoring(WIN)

            return 0

    distance_t0 = -999.0
    # 相手に向かっているときボーナス
    def closeBonus(self):
        distance_t1 = np.linalg.norm(self.players[0].pos - self.players[1].pos)
        if self.distance_t0 > distance_t1:
            for pid in range(player_count):
                if 1 <= self.actions[pid] and self.actions[pid] <= 4:
                    self.players[pid].scoring(CLOSEBONUS)
        self.distance_t0 = distance_t1

    # 出力
    def render(self, pid):
        board = copy.deepcopy(self.board)
        for x in range(boardXsize):
            for y in range(boardYsize):
                chip = int(board[x, y])
                if 10 <= chip and chip < 100:
                    if chip == pid + 10:
                        board[x, y] = ITEMS
                    # 敵性プレイヤー
                    else:
                        board[x, y] = ITEMS + 1
                # 爆弾
                elif chip == 100:
                    board[x, y] = ITEMS + 2
                # 爆弾onMe
                elif chip == 110 + pid:
                    board[x, y] = ITEMS + 3
                # 爆弾onEnemy
                elif not chip < ITEMS:
                    board[x, y] = ITEMS + 4

        return board

    def renderForHuman(self, pid, board=None):
        b = self.board
        if type(board) != type(None):
            b = board
        img = []
        for x in range(boardXsize):
            img_x = []
            for y in range(boardYsize):
                if b[x, y] == 0:
                    img_x.append(chip_none)
                elif b[x, y] == 1:
                    img_x.append(chip_block)
                elif b[x, y] == 2:
                    img_x.append(chip_brick)
                elif b[x, y] == 10 + pid:
                    img_x.append(chip_player)
                elif 10 <= b[x, y] and b[x, y] < 100:
                    img_x.append(chip_enemy)
                elif b[x, y] == 100:
                    img_x.append(chip_bomb)
                elif b[x, y] == 110 + pid:
                    img_x.append(chip_bomb_on_me)
                else:
                    img_x.append(chip_bomb_on_enemy)
            img_x = cv2.vconcat(img_x)
            img.append(img_x)
        img = cv2.hconcat(img)
        return img

    def renderToGif(self, pid=0, path="play"):
        images = []
        for board in self.boardHistory:
            images.append(
                Image.fromarray(
                    cv2.pyrUp(self.renderForHuman(pid=pid, board=board)), mode="RGB"
                )
            )
        images[0].save(
            path + ".gif", save_all=True, append_images=images[1:], duration=300, loop=0
        )

    # 以下gym用の関数

    # リワード設定
    def rewards(self):
        rets = []
        for pid in range(player_count):
            rets.append(self.players[pid].score)
        return rets

    # 観測
    def observations(self):
        rets = []
        for pid in range(player_count):
            rets.append(
                torch.Tensor(self.render(pid=pid).reshape(1, windowXsize, windowYsize))
            )
        return rets

    # アクション登録
    def addAction(self, pid, action):
        if pid < player_count and action <= n_actions:
            self.actions[pid] = action

    # 時間経過
    def step(self):
        for pid in range(player_count):
            action = self.actions[pid]
            if 1 <= action and action <= 4:
                self.move(pid, action - 1)
            elif action == 5:
                self.bombSet(pid, self.players[pid].pos)

        if not self.isFin:
            self.bombTick()
            self.boardHistory.append(copy.deepcopy(self.board))
            self.closeBonus()
            if sum(self.playableId) <= 1:
                self.isFin = True
        # 生存スコア計算
        for pid in range(player_count):
            if self.actions[pid] != 0:
                self.players[pid].scoring(ALLIVE)
            else:
                self.players[pid].scoring(INVALID_ACTION_PENALTY)

        return self.observations(), self.rewards(), self.isFin, {}
