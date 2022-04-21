import cv2
import numpy as np

# ゲーム情報定義
# ------------
# フィールドのサイズ定義　両方奇数である前提で行っている
# board_x_size = 13
# board_y_size = 11
board_x_size = 7
board_y_size = 7
# フィールドにおける爆弾の合計の上限数
MAXBOMBS = 20
ITEMS = 3
# プレイヤー人数　現状2である前提
player_count = 2
# 行動可能なアクションの数
n_actions = 6
# 最初においてあるレンガの数
brickCount = 0
# ゲームクラスのデバッグ用
logDisp = False
# 各modelの可視化用
dummyCheck = True

assert board_x_size % 2 == 1 and board_y_size % 2 == 1
assert player_count == 2

# 報酬設定
# -------
ON_BOM_HIT = 0
ON_BOM_HIT_BY = 0
ON_BOM_SELF = 0
ON_BREAK_BRICK = 0
ON_CLOSE_GAP = 0
ON_WIN = 1
ON_LOSE = -1
ON_INVALID_ACTION = 0
ON_ALIVE = 0

# ON_BOM_HIT = 10
# ON_BOM_HIT_BY = -0.01
# ON_BOM_SELF = -1
# ON_BREAK_BRICK = 0.002
# ON_CLOSE_GAP = 0.01
# ON_WIN = 0.2
# ON_LOSE = -0.2
# ON_INVALID_ACTION = -0.008
# ON_ALIVE = 0.0002

# マップ用画像読み込み
pix_size = 1
window_x_size = board_x_size * pix_size
window_y_size = board_y_size * pix_size

V = np.array([[0, -1], [0, 1], [1, 0], [-1, 0]])
