import sys
from absl import flags, app
from utils.model_3D import load3DModel, fitModel2UnitSphere
from sklearn.gaussian_process.kernels import RationalQuadratic
import numpy as np
from utils.gp_utils import findCentersKmeans
from utils.gp_utils import ClusteredGPs
from vis.vis import *

from rich.console import Console
from rich.progress import track
from utils.common import *


sys.path.append("..")

FLAGS = flags.FLAGS

flags.DEFINE_string("model_3d", "./model.ply", "Path to the 3D model in .ply format")
flags.DEFINE_integer("num_ref_points", 5, "Nummber of reference points for clustering.")
flags.DEFINE_string("centers", None, "Path to custom center coordinates")
flags.DEFINE_float("overlap", None, "Use inter-cluster overlap")
flags.DEFINE_bool("vis_overlap", False, "Visualize overlaping regions")


def main(args):

    # load points
    points = load3DModel(jn(MODEL_PATH_TRAIN,FLAGS.model_3d+'.ply'))
    np.random.shuffle(points)
    points = points[:1000]
    # scale to unit sphere
    points = fitModel2UnitSphere(points, buffer=1.0 / 1.03)

    create_directory(jn(RESULTS_PATH,FLAGS.model_3d.split('.')[0]))
    if FLAGS.centers is not None:
        centers = np.loadtxt(FLAGS.centers, delimiter=",")
    else:
        # cluster points using K-means
        labels, centers, _ = findCentersKmeans(
            points.astype(np.float64),
            clusters=FLAGS.num_ref_points,
            savePath=jn(RESULTS_PATH,FLAGS.model_3d.split('.')[0]),
        )

    # make overlaping regions
    if FLAGS.overlap is not None:
        overlap_radius_ratio = FLAGS.overlap
        overlaping_radius_per_class = []
        for idx, c in enumerate(centers):
            points_of_class = points[labels == idx]
            point_distances = np.linalg.norm(points_of_class - c, axis=-1)
            overlap_radius = (
                point_distances.max() + point_distances.max() * overlap_radius_ratio
            )
            overlaping_radius_per_class.append(overlap_radius)

        classified_points = []
        labels_mod = []

        for idx_c, c in enumerate(centers):
            points_per_class = []
            for idx, p in enumerate(points):
                # calculate the distance to each center
                distance = np.linalg.norm(p - c, axis=-1)
                if distance < overlaping_radius_per_class[idx_c]:
                    # then include this point in this class
                    points_per_class.append(p)
                    labels_mod.append(idx_c)
            # finally append the array to the list
            classified_points.append(np.array(points_per_class))

        classified_points = np.concatenate(classified_points, axis=0)
    else:
        classified_points = points
        labels_mod = labels

    if FLAGS.vis_overlap:
        u, c = np.unique(classified_points, axis=0, return_counts=True)
        dup_indices = np.isin(classified_points, u[c > 1]).all(axis=1)
        create_colored_points_renderer(
            classified_points,
            labels_mod,
            np.bincount(labels_mod),
            duplicates=dup_indices,
        )

    labels_mod = np.array(labels_mod)
    distances = distance_from_centers(classified_points, centers, labels_mod)
    _, phis_thetas_train, ds, _ = direction_distance_given_class(
        classified_points, distances, centers, labels_mod
    )

    kernel = RationalQuadratic(
        alpha=1,
        length_scale=0.1,
        length_scale_bounds=(1e-5, 1e2),
        alpha_bounds=(1e-7, 1),
    )

    cls_gp = ClusteredGPs(centers)
    cls_gp.create(kernel, n_restarts_optimizer=10)
    cls_gp.fit_no_labels(phis_thetas_train, ds)
    cls_gp.__save__(jn(RESULTS_PATH,FLAGS.model_3d.split('.')[0]))


if __name__ == "__main__":
    app.run(main)
