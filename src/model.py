class BaseModel:
    def __init__(self):
        self.model = None

    def build(self):
        raise NotImplementedError

    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
