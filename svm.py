from PIL import Image
import numpy as np
import os 
import sys
import random
from ssa_clustering import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from skimage import exposure

def build_dataset(neg, pos, trans=True, usessa=False):
    nlist = [ neg + x for x in os.listdir(neg) 
            if os.path.isfile(neg + x) and x.endswith("jpg")]
    plist = [ pos + x for x in os.listdir(pos)
            if os.path.isfile(pos + x) and x.endswith("jpg")]

    nims = [ np.asarray(Image.open(im)).flatten() for im in nlist]
    pims = [ np.asarray(Image.open(im)).flatten() for im in plist]
    if usessa:
        nims = []
        for im in nlist:
            r, pr, r_max = ssa_clustering(im, 40, 2, 0, 0.1, False, 42)
            nims.append(exposure.equalize_hist(np.asarray(r_max)).flatten())

            if len(nims) % 10 == 0:
                print("Progress: {0}/{1}...".format(len(nims), len(nlist)))
        print("...Negative samples processed!")
        pims = []
        for im in plist:
            r, pr, r_max = ssa_clustering(im, 40, 2, 3, 0.25, False, 42)
            pims.append(exposure.equalize_hist(np.asarray(r_max)).flatten())
            if len(pims) % 10 == 0:
                print("Progress: {0}/{1}...".format(len(pims), len(plist)))
        print("...Positive samples processed!")
    rate = float(len(nims)) / len(pims)
    dataset = [(x, 0) for x in nims] + [(x, 1) for x in pims] 
    return dataset, rate

if __name__ == "__main__":
    NEGATIVE_DIR = sys.argv[1]
    POSITIVE_DIR = sys.argv[2]

    dataset, rate = build_dataset(NEGATIVE_DIR, POSITIVE_DIR, usessa=True)
    l = len(dataset)
    print("...Dataset building complepted.")
    print("""
    -------------------------------
        Dataset Size: {0}
        Negative/Positvie: {1:4.2f}
    -------------------------------
          """.format(l, rate))
    seed = 42
    random.seed(seed)
    random.shuffle(dataset)
    scaler = StandardScaler()
    T = int(0.7 * l)
    X = scaler.fit_transform([x[0] for x in dataset[:T]])
    y = np.asarray([x[1] for x in dataset[:T]])
    X_test = scaler.transform([x[0] for x in dataset[T:]])
    y_test = np.asarray([x[1] for x in dataset[T:]])
    lsvc = SVC(C=10, gamma=0.0001,kernel='rbf', class_weight='balanced').fit(X, y)
    acc = accuracy_score(lsvc.predict(X_test), y_test)
    print(acc)
    cross_validation=False
if cross_validation:
        X = scaler.fit_transform(np.asarray([x[0] for x in dataset]))
        y = np.asarray([x[1] for x in dataset])
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(class_weight='balanced'), param_grid=param_grid, cv=cv)
        grid.fit(X, y)
    
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
