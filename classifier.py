import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps 

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).values_counts())

classess = ['A', 'B', 'C', 'D', 'E' 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size = 3500, test_size = 500, random_state = 9)

xtrainscaled = xtrain/255.0
xtestscaled = xtest/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(xtrainscaled, ytrain)

def getPrediction(image):
    im_pil = Image.open(image)
    imagebw = im_pil.convert("L")
    imagebwresized = imagebw.resize((22, 30), Image.ANTIALIAS)

    pixelfilter = 20
    minpixel = np.percentile(imagebwresized, pixelfilter)
    imagebwresizedinvertedscaled = np.clip(imagebwresized-minpixel, 0, 255)
    maxpixel = np.max(imagebwresized)
    imagebwresizedinvertedscaled = np.asarray(imagebwresizedinvertedscaled)/maxpixel

    testsample = np.array(imagebwresizedinvertedscaled).reshape(1, 660)
    testpred = clf.predict(testsample)

    return testpred[0]