class Bomb:
    def __init__(self):
        self.isActive = False
        self.timer = 8
        self.who = -1
        self.len = -1
        self.pos = [-1, -1]

    def activate(self, pos, id, len):
        self.pos = pos
        self.who = id
        self.len = len
        self.isActive = True
