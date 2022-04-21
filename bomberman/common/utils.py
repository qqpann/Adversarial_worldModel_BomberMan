import numpy as np

from .variables import board_x_size, board_y_size


# パラメータ、便利関数
# -----------------
def pos2v(val):
    return val[0], val[1]


def nearInt(val):
    return int(val + 0.5)


def isInBoard(p):
    if 0 <= p[0] and p[0] < board_x_size and 0 <= p[1] and p[1] < board_y_size:
        return True
    else:
        return False


def rand_ints_nodup(a, b, k):
    ns = []
    if k > b:
        k = b
    while len(ns) < k:
        n = np.random.randint(a, b)
        if not n in ns:
            ns.append(n)
    return ns


# # renderForHumanと同義
# def matToImg(mat, fromFloat=False, pid=0):
#     mat = np.array(copy.deepcopy(mat).reshape(windowXsize, windowYsize))
#     # 学習用に[0,1]の範囲にされたものをゲーム処理用に直す
#     if fromFloat:
#         mat *= ITEMS + 4
#         for x in range(boardXsize):
#             for y in range(boardYsize):
#                 chip = nearInt(mat[x, y])
#                 if chip == ITEMS:
#                     mat[x, y] = 10
#                 elif chip == ITEMS + 1:
#                     mat[x, y] = 11
#                 elif chip == ITEMS + 2:
#                     mat[x, y] = 100
#                 elif chip == ITEMS + 3:
#                     mat[x, y] = 110
#                 elif not chip < ITEMS:
#                     mat[x, y] = 111
#                 else:
#                     mat[x, y] = chip
#     img = []
#     # ゲーム処理用の配列をEnv()のインスタンスがないためここで人間用の画像に処理
#     for x in range(boardXsize):
#         img_x = []
#         for y in range(boardYsize):
#             if mat[x, y] == 0:
#                 img_x.append(chip_none)
#             elif mat[x, y] == 1:
#                 img_x.append(chip_block)
#             elif mat[x, y] == 2:
#                 img_x.append(chip_brick)
#             elif mat[x, y] == 10 + pid:
#                 img_x.append(chip_player)
#             elif 10 <= mat[x, y] and mat[x, y] < 100:
#                 img_x.append(chip_enemy)
#             elif mat[x, y] == 100:
#                 img_x.append(chip_bomb)
#             elif mat[x, y] == 110 + pid:
#                 img_x.append(chip_bomb_on_me)
#             else:
#                 img_x.append(chip_bomb_on_enemy)
#         img_x = cv2.vconcat(img_x)
#         img.append(img_x)
#     img = cv2.hconcat(img)
#     return img
