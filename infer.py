import sys
from absl import flags, app, logging
from utils.model_3D import *
import numpy as np
import pickle
from os.path import join as j
from utils.metrics import chamfer_distance, mean_p2p_distance
from utils.gp_utils import (
    direction_distance_given_class,
    distance_from_centers,
    ClusteredGPs,
)
from utils.common import *
from vis.vis import *

from rich.progress import track
from rich.console import Console

sys.path.append("..")

FLAGS = flags.FLAGS



flags.DEFINE_bool("vis", False, "Visualize gp output or not")
flags.DEFINE_string("model_3d", None, "Test model 3D")


def main(args):

    # load the test points
    points = load3DModel(jn(MODEL_PATH_TEST,FLAGS.model_3d+'.ply'))
    np.random.shuffle(points)

    # scale to unit sphere
    #points = fitModel2UnitSphere(points, buffer=1.03)

    # load the fitted kmeans centers
    with open(jn(RESULTS_PATH,FLAGS.model_3d,'kmeans.pkl'), "rb") as f:
        kmeans = pickle.load(f)
    centers = kmeans.cluster_centers_
    class_idxs = kmeans.predict(points.astype("double"))

    # get distance each point from the reference points
    distances = distance_from_centers(points, centers, class_idxs)
    _, phi_thetas, ds_observed, sorted_indices = direction_distance_given_class(
        points, distances, centers, class_idxs
    )

    # load the trained GP's
    cls_gps = ClusteredGPs(centers)
    cls_gps.__load__(jn(RESULTS_PATH,FLAGS.model_3d))
    _, xyz, _ = cls_gps.predict_(phi_thetas, centers, class_idxs)
    cls_gps.eval(points[sorted_indices], xyz)

    export_3D_points(xyz, jn(RESULTS_PATH,FLAGS.model_3d,"out_points.txt"))

    if FLAGS.vis:
        vis_pcd_open3D(xyz, jn(RESULTS_PATH,"./test.png"))


if __name__ == "__main__":
    app.run(main)
