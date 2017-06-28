from PIL import Image
from rssa import *
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
def ssa_clustering(img, L, c, cut, factor, norm, seed):
    im = np.asarray(Image.open(img)) / 256.0
    num =int(L*L * factor)
    s, u = ssa(im, L)
    if norm:
        u = u / np.linalg.norm(u, axis=0)
    u = u[cut:num]
    print(u[0])
    print(u[1])
    print(np.dot(u[0], u[1]))
    kmeans = KMeans(n_clusters=c, random_state=seed, n_jobs=-1).fit(u)
    groups = []
    for i in range(c):
        r = []
        for j in range(len(kmeans.labels_)):
            if i == kmeans.labels_[j]:
                r.append(j+1+cut)
        groups.append(r)
    r = [ im * 256.0 for im in reconstruct(s, groups)]
    pr = [im * 256.0] + r + [(im*256.0 - np.sum(r, axis=0))]
    r_max = max(r, key=lambda im: np.var(im))
    return r, pr, r_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decompse a image and reconsturct it with 2d-ssa')
    parser.add_argument('--img', required=True, help="The path to img")
    parser.add_argument('--norm', action='store_true', help="Normalize eigenvectors")
    parser.add_argument('--cut', default=0, type=int,help="Cut some first eigen vectors.")
    parser.add_argument('--L', default=40,type=int, help="Window size")
    parser.add_argument('--c', default=3, type=int, help="Number of classes")
    parser.add_argument('--seed', default=7, type=int, help="Random seed")
    parser.add_argument('--factor', default=0.25, type=float, help="factor * L * L== number of eigenvector to clustring")
    opt = parser.parse_args()
    print(opt)
    img = opt.img
    cut = opt.cut
    norm = opt.norm
    L = opt.L
    c = opt.c
    seed = opt.seed
    factor = opt.factor
    r, pr, r_max = ssa_clustering(img, L, c, cut, factor, norm, seed) 

    fig, axes = plt.subplots(1,c+2)
    for ax, img in zip(axes, pr):
        ax.imshow(img, cmap=plt.cm.gray)
        v = np.var(img)
        ax.set_title("Var: {0:4.2f}".format(v) )
    plt.show()
