import numpy as np
from skimage import feature


class LocalBinaryPatterns:
    def __init__(self, numPoints, raio):
        self.numPoints = numPoints
        self.raio = raio

    def describe(self, image, eps=1e-7):
        # calcula a representação LBP da imagem e depois usa isso para construir o
        # histograma de padrões
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.raio, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normaliza o histograma
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist
