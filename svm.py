from PIL import Image
import numpy as np
import os 
import sys
import random
from ssa_clustering import *
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from skimage import exposure

def build_dataset(neg, pos, trans=True, usessa=False, argue=False):
    nlist = [ neg + x for x in os.listdir(neg) 
            if os.path.isfile(neg + x) and x.endswith("jpg")]
    plist = [ pos + x for x in os.listdir(pos)
            if os.path.isfile(pos + x) and x.endswith("jpg")]

    nims = [ np.asarray(Image.open(im)).flatten() for im in nlist]
    pims = [ np.asarray(Image.open(im)).flatten() for im in plist]
    if argue:
        seed = 42
        random.seed(seed)
        dataset = [(x, 0) for x in nlist] + [(x, 1) for x in plist]
        rate = float(len(nlist)) / len(plist)
        random.shuffle(dataset)
        T = int(0.7 * len(dataset))
        train_set = []
        for im, label in dataset[:T]:
            img = Image.open(im)
#            img.rotate(90).show()
            train_set.append((np.asarray(img).flatten(), label))
            train_set.append((np.asarray(img.rotate(90)).flatten(), label))
            train_set.append((np.asarray(img.rotate(180)).flatten(), label))
            train_set.append((np.asarray(img.rotate(270)).flatten(), label))
        test_set = []
        for im, label in dataset[T:]:
            img = Image.open(im)
            test_set.append((np.asarray(img).flatten(), label))
        return train_set, test_set, rate
    if usessa:
        nims = []
        for im in nlist:
            r, pr, r_max = ssa_clustering(im, 40, 2, 1, 0.2, False, 42)
            nims.append(exposure.equalize_hist(np.asarray(r_max)).flatten())

            if len(nims) % 10 == 0:
                print("Progress: {0}/{1}...".format(len(nims), len(nlist)))
        print("...Negative samples processed!")
        pims = []
        for im in plist:
            r, pr, r_max = ssa_clustering(im, 40, 2, 1, 0.2, False, 42)
            pims.append(exposure.equalize_hist(np.asarray(r_max)).flatten())
            if len(pims) % 10 == 0:
                print("Progress: {0}/{1}...".format(len(pims), len(plist)))
        print("...Positive samples processed!")
    rate = float(len(nims)) / len(pims)
    dataset = [(x, 0) for x in nims] + [(x, 1) for x in pims]
    return dataset, rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decompse a image and reconsturct it with 2d-ssa than do the classfication ')
    parser.add_argument('--NEGATIVE_DIR', required=True, help="The path to neg_img")
    parser.add_argument('--POSITIVE_DIR', required=True, help="The path to posi_img")
    parser.add_argument('--usessa', action='store_true', help="Ust the ssa")
    parser.add_argument('--argue', action='store_true', help="arguement of dataset")
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    opt = parser.parse_args()

    NEGATIVE_DIR = opt.NEGATIVE_DIR
    POSITIVE_DIR = opt.POSITIVE_DIR

    if opt.argue:
        train_set, test_set, rate = build_dataset(NEGATIVE_DIR, POSITIVE_DIR, argue=opt.argue)
        l = len(train_set)
        print("...Dataset building complepted.")
        print("""
    -------------------------------
        Dataset Size: {0}
        Negative/Positvie: {1:4.2f}
    -------------------------------
          """.format(l, rate))
        scaler = StandardScaler()
        X = scaler.fit_transform([x[0] for x in train_set])
        y = np.asarray([x[1] for x in train_set])
        X_test = scaler.transform([x[0] for x in test_set])
        y_test = np.asarray([x[1] for x in test_set])
    else:        
        dataset, rate = build_dataset(NEGATIVE_DIR, POSITIVE_DIR, usessa=opt.usessa)
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
    if opt.linear:
        lsvc = LinearSVC(C=10,class_weight='balanced').fit(X,y)
    else:
        lsvc = SVC(C=10, gamma=0.0001,kernel='rbf', class_weight='balanced').fit(X, y)
    acc = accuracy_score(lsvc.predict(X_test), y_test)
    print(acc)
    cross_validation=opt.grid_search
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
