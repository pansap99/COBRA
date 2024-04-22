import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

def ADD(modelPoints,Rgt,Rest,Tgt,Test):

    gts = Rgt[None,:].dot(modelPoints.T) + Tgt.reshape((3,1))
    ests = Rest[None,:].dot(modelPoints.T) + Test.reshape((3,1))
    add = np.linalg.norm(ests.T-gts.T,axis=1).mean()
   
    return add

def normalize_error(error,sizes):



    return error / max(*sizes)

def chamfer_distance(p1,p2):
    
    tree = KDTree(p1)
    dist_p1 = tree.query(p2)[0]
    tree = KDTree(p2)
    dist_p2 = tree.query(p1)[0]
    
    return dist_p1.mean() + dist_p2.mean()

def mean_p2p_distance(p1,p2,norm=2):
    
    # calculate the L2 norm of each point
    distances = np.linalg.norm(p1-p2,axis=-1)
    
    return distances.mean(),np.median(distances),distances.std()

def evaluate_predicted_3D_points(points):
    pass