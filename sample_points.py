import vtk
import math as mt
import numpy as np
import sys
import time

sys.path.append("..")
from utils.model_3D import *
from utils.gp_utils import *
from vis.vis import *
from absl import flags, app
import glob
from utils.common import *

FLAGS = flags.FLAGS


flags.DEFINE_string("modelPath", None, "Test model 3D")
flags.DEFINE_enum(
    "method", "poisson", ["poisson", "ray-casting"], "Method for sampling points"
)
flags.DEFINE_integer(
    "num_points_poisson", None, "Num points to generate with poisson sampling."
)
flags.DEFINE_integer("num_ref_points", None, "Num ref points for raycasting.")
flags.DEFINE_string("savePath", "./centers.txt", "Txt file with the center coordinates")
flags.DEFINE_float("step",1.5, "Phi, theta sampling steps")
flags.DEFINE_integer('random_seed',42,'Random seed')


def main(args):
    
    for model in glob.glob(FLAGS.modelPath + "/*.ply"):
        reader = vtk.vtkPLYReader()
        reader.SetFileName(model)
        reader.Update()
        poly_data = reader.GetOutput()
        vertices = load3DModel(model)

        time_start = time.time()

        if FLAGS.method == "ray-casting":
            _, centers, _ = findCentersKmeans(vertices, FLAGS.num_ref_points)
            ppoints, class_idxs = ray_casting_from_centers(
                np.array([0, 0, 0]),
                centers,
                load3DModel(model),
                poly_data,
                savePath=FLAGS.savePath,
                step=FLAGS.step,
                savePerClass=False,
                mode="min",
            )
            ppoints = np.concatenate(ppoints, axis=0)
            writePLY(
                [tuple(point) for point in ppoints.tolist()],
                os.path.join(FLAGS.savePath, "test.ply"),
            )
            vis_points_o3d(
                ppoints,
                center_points=centers,
                sphere_radius=vertices.max(axis=-1).max() * 0.02,
            )

            print("Time : ", time.time() - time_start, " Total points: ", len(ppoints))

        elif FLAGS.method == "poisson":

            input_mesh = o3d.io.read_triangle_mesh(model)
            np.random.seed(42)
            point_cloud = input_mesh.sample_points_poisson_disk(
                number_of_points=FLAGS.num_points_poisson
            )
            points = np.array(point_cloud.points)
            writePLY(
                [tuple(point) for point in points.tolist()],
                os.path.join(
                    FLAGS.savePath, os.path.basename(model)),
            )


if __name__ == "__main__":
    app.run(main)
