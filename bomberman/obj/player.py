import numpy as np


class Player:
    def __init__(self, id=0, debug=False):
        self.id = id
        self.life = 1
        self.pos = np.array([0, 0])
        self.bombMax = 2
        self.setBomb = 0
        self.bombLen = 15
        self.score = 0.0
        self.debug = debug

    def _changePos(self, pos):
        self.pos = pos

    def scoring(self, score):
        self.score += score

    def dmg(self):
        self.life -= 1
        if self.debug:
            print(f"{self.id}の残機減少：残り{self.life}")

    def isSurvive(self):
        if self.life > 0:
            return True
        else:
            return False
