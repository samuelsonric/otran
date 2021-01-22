from numpy.random import choice
from numpy import clip
from matplotlib.pyplot import imread
from ot.da import EMDTransport

def sample_rows(X, n):
    return X[choice(len(X), n)]

def read_image(path):
    return imread(path).astype(float) / 256

class ColorTransfer:
    def __init__(self, *paths):
        self.im = tuple(map(read_image, paths))
        self.flat = tuple(map(lambda x: x.reshape(-1, 3), self.im))
        self.model = EMDTransport()

    def fit(self, n):
        Xs, Xt = tuple(map(lambda x : sample_rows(x, n), self.flat))
        self.model.fit(Xs = Xs, Xt = Xt)
        return self

    def transform(self):
        im = self.model.transform(Xs = self.flat[0])
        return clip(im, 0, 1).reshape(*self.im[0].shape)


