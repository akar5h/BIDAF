class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        new_avg = (1- self.mu)*x + self.mu * self.shadow[name]
        self.shadow[name] = new_avg.clone()

