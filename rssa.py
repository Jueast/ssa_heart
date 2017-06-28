import rpy2.robjects
from rpy2.robjects.packages import importr
import numpy as np
import rpy2.robjects.numpy2ri

R = rpy2.robjects.r
Rssa = importr('Rssa')
rpy2.robjects.numpy2ri.activate()
base = importr("base")
dollar = base.__dict__["$"]
# im should be numpy array L x W
# L should be a list of window size
# this function will return a dict to reconstruction to different window size
def ssa_reconstruct(im, L):
    result = {}
    for l in L:
        s = Rssa.ssa(im, kind="2d-ssa", L=R("c({0},{0})".format(str(l))))
        temp = [str(i) for i in range(1, int((l*l)/2))]
        groups = R("list({0})".format(",".join(temp)))
        r = Rssa.reconstruct(s, groups=groups)
        sr = [np.asarray(x) for x in r]
        result[l] = sr
    return result

def ssa(im, L):
    s = Rssa.ssa(im, kind="2d-ssa", L=R("c({0},{0})".format(L)))
    return s, np.asarray(dollar(s,"U"))

def wcor(s, num):
    return np.asarray(Rssa.wcor(s, groups=R("1:{0}".format(num))))

def reconstruct(s, groups):
    groups = [ (str(i), np.asarray(groups[i])) for i in range(len(groups))]
    rgroups = rpy2.robjects.ListVector(dict(groups))
    r = Rssa.reconstruct(s, groups=rgroups)
    sr = [np.asarray(x) for x in r]
    return sr
    
