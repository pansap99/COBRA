import sys
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
import pickle
import math as mt
import os
from utils.model_3D import spherical_coordinates_to_cartesian, export_3D_points
from sklearn.cluster import KMeans
from utils.metrics import *
from utils.io import create_directory
import vtk
from rich.table import Table
from typing import Literal
from utils.model_3D import writePLY, clustering_decimation, load3DModel

import time
from numpy.typing import NDArray
from rich.console import Console

sys.path.append("..")

CONSOLE = Console()


def cartesian2spherical(point):

    distance = mt.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
    phi = np.arctan2(point[1], point[0])
    theta = np.arccos(point[2] / (distance))

    return phi, theta, distance


def sort_class_indices(class_idxs, points_per_class):

    new_indcies = np.zeros((len(class_idxs), 1))
    cumulative_class_points = np.cumsum(points_per_class)

    start_indices = np.concatenate(([0], cumulative_class_points[:-1]))
    end_indices = cumulative_class_points

    for i in range(len(points_per_class)):
        new_indcies[start_indices[i] : end_indices[i]] = i

    return new_indcies.flatten().astype(np.int32).tolist()


def distance_from_centers(points, centers, class_idxs):
    # reshape to use numpy broadcasting
    points = points[:, None, :]
    centers = centers[None, :, :]

    distances = np.linalg.norm(points - centers, axis=-1)

    return distances[np.arange(len(distances)), class_idxs]


def classify_points_to_center(points: NDArray, centers: NDArray) -> NDArray:
    """Given an array of points of shape (n,3) and an array of center coordinates
    of shape (c,3) it returns the indices of the closest center to each point.

    Args:
        points (NDArray): Input points array.
        centers (NDArray): Numpy array with 3D center locations of shape (N,3).

    Returns:
        point_distances (NDArray): Distances of each point to the closest center.
        cls_center_idx (NDArray): Array with closest center indices.
    """
    # reshape to use numpy broadcasting
    points = points[:, None, :]
    centers = centers[None, :, :]

    # assign the point to its closest center
    closest_center = np.linalg.norm(points - centers, axis=-1)
    cls_center_idx = np.argmin(closest_center, axis=1)

    # Create an array of distances from each point to its closest center
    point_distances = closest_center[np.arange(len(closest_center)), cls_center_idx]

    return point_distances, cls_center_idx


def point3D_to_direction(points3D: NDArray, centers: NDArray, class_idxs: NDArray):
    """Given some 3D points that have been already classified with some classifier
    they are converted into direction (phi,theta) and distance (d) w.r.t to their assigned center.

    The returned arrays are not sorted by class and instead have the same indexing as the
    input 3D points.

    Args:
        points3D (NDArray): 3D points to parameterize as direction and distance.
        centers (NDArray): 3D locations of centers
        class_idxs (NDArray): Array with the class indices of the classified 3D points.
        of shape (N,).

    Returns:
        phi_thetas (NDArray): Array of shape (N,2) with directions.
        ds (NDArray): Array of shape (N,1) with distances.
    """

    # Transform each point using the 3D coordinates of the assigned center
    points = points3D - centers[class_idxs]
    distances = np.linalg.norm(points, axis=-1)

    # Compute the spherical coordinates of the transformed points
    # w.r.t to their assigned center
    phis = np.arctan2(points[:, 1], points[:, 0])  # polar angle
    thetas = np.arccos(points[:, 2] / (distances))  # azimuth

    return np.column_stack((phis, thetas)), distances


def direction_distance_given_class(
    points,
    distances,
    centers,
    cls_center_idxs,
    saveClassPointsPath=None,
    return_scaled=False,
):

    clusters = []
    phi_thetas = []
    ds = []
    scalers = []
    cluster_indices = []
    unique_clusters = np.unique(cls_center_idxs)
    points = points - centers[cls_center_idxs]

    if saveClassPointsPath:
        f = open(os.path.join(saveClassPointsPath, "infer_classes.txt"), "w")
    for cluster in range(len(centers)):
        indices_for_cluster_i = np.where(cls_center_idxs==cluster)[0]
        cluster_points = points[indices_for_cluster_i]

        if len(cluster_points) == 0:
            phi_thetas.append([])
            ds.append([])
        else:
            if saveClassPointsPath:
                f.write(f"Class {cluster}: {len(cluster_points)}\n")
            clusters.append(cluster_points)

            phis = np.arctan2(cluster_points[:, 1], cluster_points[:, 0])  # polar angle
            thetas = np.arccos(
                cluster_points[:, 2] / (distances[indices_for_cluster_i])
            )  # azimuth

            if return_scaled:
                phis /= np.pi
                thetas /= 2 * np.pi
                distances[indices_for_cluster_i] /= distances.max()

            phi_thetas.append(np.column_stack((phis, thetas)))
            ds.append(distances[indices_for_cluster_i])
            cluster_indices.append(indices_for_cluster_i)

    if saveClassPointsPath:
        f.close()
    return clusters, phi_thetas, ds, np.concatenate(cluster_indices)


def xyz_from_direction_distance_class_signle(phi_theta, ds, centers, class_idx):
    xyz = []
    x, y, z = spherical_coordinates_to_cartesian(phi_theta[0], phi_theta[1], ds)
    xyz = np.column_stack((x, y, z)) + centers[class_idx]

    return np.array(xyz)


def xyz_from_direction_distance_class(phi_theta, ds, centers, class_idx):

    xyz = []
    x, y, z = spherical_coordinates_to_cartesian(phi_theta[:, 0], phi_theta[:, 1], ds)
    xyz = np.column_stack((x, y, z)) + centers[class_idx]

    return np.array(xyz)


def findCentersKmeans(points, clusters, savePath=None):

    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(points)
    if savePath is not None:
        with open(os.path.join(savePath, "kmeans.pkl"), "wb") as f:
            pickle.dump(kmeans, f)
        np.savetxt(
            os.path.join(savePath, "kmeans_centers.txt"),
            kmeans.cluster_centers_,
            delimiter=",",
        )

    return kmeans.labels_, kmeans.cluster_centers_, kmeans


def ray_casting_layers(
    center,
    model_3D,
    polydata,
    step=[5, 5],
    keep: Literal["all", "min", "max", "split_all"] = "all",
):

    # calculate sampling ray radius
    max_dims = np.max(model_3D, axis=0)
    sampling_radius = 2 * np.linalg.norm(max_dims - center)

    # calculate intersections
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(polydata)
    obb_tree.BuildLocator()

    pointsVTKintersection = vtk.vtkPoints()
    pointsIntersection = []
    max_intersections = 0
    # get sampling point coordinates with phi,theta with given step and sampling radius
    for phi in np.arange(0, 360, step[0]):
        for theta in np.arange(0, 180, step[1]):
            # print(phi,theta)
            pTarget = spherical_coordinates_to_cartesian(
                np.radians(phi), np.radians(theta), sampling_radius
            )
            code = obb_tree.IntersectWithLine(
                center, pTarget, pointsVTKintersection, None
            )
            pointsVTKIntersectionData = pointsVTKintersection.GetData()
            noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

            if max_intersections < noPointsVTKIntersection:
                max_intersections = noPointsVTKIntersection

            distances2center = []
            points = []
            for idx in range(noPointsVTKIntersection):
                _tup = pointsVTKIntersectionData.GetTuple3(idx)
                distances2center.append(np.linalg.norm(_tup - center))
                points.append(_tup)
                # pointsIntersection.append(_tup)
            points = np.array(points)
            if len(distances2center) >= 1:
                pointsIntersection.append(
                    points[np.argsort(np.array(distances2center))]
                )
    gps = {}
    for p in pointsIntersection:
        p = p.tolist()
        key_len = len(p)
        # add the point to the gp id equal to the number of intersections of the ray
        for i in range(key_len):
            # Check if the length of p is already a key in gps
            if i in gps:
                gps[i].append(p[i])
            else:
                gps[i] = [p[i]]

        # also add this point to all gps > point intersection
        for i in range(len(p), max_intersections):
            # Check if the length i is already a key in gps
            if i in gps:
                gps[i].append(p[len(p) - 1])
            else:
                gps[i] = [p[len(p) - 1]]

    return gps, pointsIntersection


def sampleItersectionPoints(
    center,
    model_3D,
    polydata,
    step=[5, 5],
    keep: Literal["all", "min", "max", "split_all"] = "all",
):

    # calculate sampling ray radius
    max_dims = np.max(model_3D, axis=0)
    sampling_radius = 2 * np.linalg.norm(max_dims - center)

    # calculate intersections
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(polydata)
    obb_tree.BuildLocator()

    pointsVTKintersection = vtk.vtkPoints()
    pointsIntersection = []
    max_intersections = 0
    # get sampling point coordinates with phi,theta with given step and sampling radius
    for phi in np.arange(0, 360, step[0]):
        for theta in np.arange(0, 180, step[1]):
            # print(phi,theta)
            pTarget = spherical_coordinates_to_cartesian(
                np.radians(phi), np.radians(theta), sampling_radius
            )
            code = obb_tree.IntersectWithLine(
                center, pTarget, pointsVTKintersection, None
            )
            pointsVTKIntersectionData = pointsVTKintersection.GetData()
            noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()

            if max_intersections < noPointsVTKIntersection:
                max_intersections = noPointsVTKIntersection

            distances2center = []
            points = []
            if noPointsVTKIntersection != 0:
                for idx in range(noPointsVTKIntersection):
                    _tup = pointsVTKIntersectionData.GetTuple3(idx)
                    distances2center.append(np.linalg.norm(_tup - center))
                    points.append(_tup)
                    # pointsIntersection.append(_tup)
                points = np.array(points)
                if keep == "all":
                    pointsIntersection.append(points)
                elif keep == "min":
                    pointsIntersection.append(
                        points[np.array(distances2center).argmin()]
                    )
                elif keep == "max":
                    pointsIntersection.append(
                        points[np.array(distances2center).argmax()]
                    )
                elif keep == "split_all":
                    if len(distances2center) >= 1:
                        pointsIntersection.append(
                            points[np.argsort(np.array(distances2center))]
                        )
    return pointsIntersection, max_intersections


def ray_casting_from_centers(
    m_origin,
    centers,
    model_3D,
    polydata,
    savePath,
    step=[5, 5],
    savePerClass=False,
    mode="all",
):

    # calculate sampling ray radius
    max_dims = np.max(model_3D, axis=0)
    sampling_radius = 2 * np.linalg.norm(max_dims - m_origin)

    # calculate intersections
    obb_tree = vtk.vtkOBBTree()
    obb_tree.SetDataSet(polydata)
    obb_tree.BuildLocator()

    pointsVTKintersection = vtk.vtkPoints()
    pointsIntersection = []
    final_points = []
    class_idxs = []

    with CONSOLE.status(
        f"[bold][orange] Ray casting for each center...", spinner="aesthetic"
    ):
        for c in range(len(centers)):
            cclass = 0
            for phi in np.arange(0, 360, step[0]):
                for theta in np.arange(0, 180, step[1]):
                    pTarget = spherical_coordinates_to_cartesian(
                        np.radians(phi), np.radians(theta), sampling_radius
                    )
                    ode = obb_tree.IntersectWithLine(
                        centers[c], pTarget, pointsVTKintersection, None
                    )
                    pointsVTKIntersectionData = pointsVTKintersection.GetData()
                    noPointsVTKIntersection = (
                        pointsVTKIntersectionData.GetNumberOfTuples()
                    )

                    # keep only the first point (first intersection) -> direct line iof sight
                    distances2center = []
                    points = []
                    if noPointsVTKIntersection != 0:
                        for idx in range(noPointsVTKIntersection):
                            _tup = pointsVTKIntersectionData.GetTuple3(idx)
                            distances2center.append(np.linalg.norm(_tup - centers[c]))
                            points.append(_tup)
                            if mode == "all":
                                pointsIntersection.append(_tup)
                        if mode == "max":
                            pointsIntersection.append(
                                points[np.array(distances2center).argmax()]
                            )
                        elif mode == "min":
                            pointsIntersection.append(
                                points[np.array(distances2center).argmin()]
                            )
            writePLY(
                pointsIntersection, os.path.join(savePath, f"class_{c}_points.ply")
            )
            fmodel = load3DModel(os.path.join(savePath, f"class_{c}_points.ply"))
            final_points.append(fmodel)
            pointsIntersection = []
            for i in range(len(fmodel)):
                class_idxs.append(c)
            os.remove(os.path.join(savePath, f"class_{c}_points.ply"))

            if savePerClass:
                writePLY(
                    pointsIntersection, os.path.join(savePath, f"class_{c}_points.ply")
                )
                clustering_decimation(
                    os.path.join(savePath, f"class_{c}_points.ply"),
                    os.path.join(savePath, f"class_{c}_points_dec.ply"),
                )
                os.remove(os.path.join(savePath, f"class_{c}_points.ply"))
                pointsIntersection = []
            CONSOLE.log(
                f"[bold green]:white_check_mark: Done ray-casting for class {c} with {len(fmodel)} points."
            )

    return final_points, np.array(class_idxs).astype(np.int16)


def addPoint(renderer, points_cords, scale=0.1, color=(1.0, 1.0, 1.0)):
    """Add a point to the given VTK renderer."""

    # Create a vtkSphereSource to represent the point
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(points_cords)
    sphere_source.SetRadius(scale)

    # Create a vtkPolyDataMapper and a vtkActor
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()

    mapper.SetInputConnection(sphere_source.GetOutputPort())
    actor.SetMapper(mapper)

    # Set the color for the actor
    actor.GetProperty().SetColor(color)

    # Add the actor to the renderer
    renderer.AddActor(actor)


class ClusteredGPs:

    def __init__(self, centers: NDArray) -> None:

        # instantiate as mamny GaussianProcessRegressor as the number of classes
        self.gps = []
        self.points_per_class = []
        self.centers = centers

    def create(self, global_kernel: Kernel, n_restarts_optimizer=10):
        self.kernel = global_kernel
        for i in range(len(self.centers)):
            self.gps.append(
                GaussianProcessRegressor(
                    global_kernel, n_restarts_optimizer=n_restarts_optimizer
                )
            )

    def fit(self, x, y, center_idxs):
        """Fit function that assumes that the points are classified using a classifier and an
        array of class indices is provided.

        Args:
            x (NDArray): Array of concatenated phi,thetas for all classes
            y (NDArray): Target distances
            center_idxs (NDArrat): Array of class labels.
        """

        for i in range(len(self.gps)):

            with CONSOLE.status(f"[bold][yellow] Fitting GPs...", spinner="aesthetic"):
                self.gps[i].fit(x[center_idxs == i], y[center_idxs == i])
            self.points_per_class.append(len(x[center_idxs == i]))
            CONSOLE.log(
                f"[bold green]:white_check_mark: Done fitting GP of class {i} with {len(x[center_idxs==i])} assigned points."
            )

    def fit_no_labels(self, x, y):
        """Fit function that assumes the input phi,thetas are structured in a ndarray with shape (c,n,2).
        Use this function if you customly set which points belong to each class.

        Args:
            x (NDArray): Array of concatenated phi,thetas for all classes of shape (c,n,2).
            Where c is the number of classes, n is the number of points of each class and 2 corresponf to the two
            direction parameters phi,theta.
            y (NDArray): Array of target distances of shape (c,n,1)
        """
        for i in range(len(self.gps)):
            with CONSOLE.status(f"[bold][yellow] Fitting GPs...", spinner="aesthetic"):
                self.gps[i].fit(x[i], y[i])
            self.points_per_class.append(len(x[i]))
            CONSOLE.log(
                f"[bold green]:white_check_mark: Done fitting GP of class {i} with {len(x[i])} assigned points."
            )

    def __save__(self, path):
        # get global kernel params
        create_directory(os.path.join(path, "gps"))
        for idx, gp in enumerate(self.gps):
            with open(os.path.join(path, "gps", f"{idx}.pkl"), "wb") as f:
                print(f"Saving gp_{idx}_params.pkl")
                pickle.dump(gp, f)
        with open(os.path.join(path, "training_points_per_class.txt"), "w") as f:
            for cls in range(len(self.centers)):
                f.write(f"Class {cls}: {self.points_per_class[cls]}\n")
        # self.gps = []

    def __load__(self, path):

        for i in range(0, len(self.centers)):
            filename = os.path.join(path, f"gps/{i}.pkl")
            with open(filename, "rb") as f:
                print(f"Loading {filename}...")
                self.gps.append(pickle.load(f))

    def predict(self, x: NDArray, centers: NDArray, class_idxs: NDArray):
        """Predict function for the trained Gaussian Processes. Given the
        input points in phi,theta, the center corrdinates and the array of the
        class indices of the points predicts the distance d using the template.

        Finally, it return the predicted distsances, the correspomnding "predicted"
        3D points and the standard deviation of each prediction.

        Args:
            x (NDArray): Array of shape (num_points,2) containing the phi,thetas
            of the observed points.
            centers (NDArray): Array of shape (num_centers,3) containing the 3D
            positions of the classification centers.
            class_idxs (NDArray): Array of shape (num_points,) containing the class indices
            of all the points w.r.t to the class centers.

        Returns:
            dps (NDArray): Array with the predicted distances
            xyz (NDArray): Array with the computed 3D points (XYZ) using
            the givven phi,theta and the predicted d.
            sigmas (NDArray): Array with the standard deviation of each predicted
            distance.
        """
        
        self.xyz, self.sigmas, self.dps = [], [], []
        for idx, direction in enumerate(x):
            # call the gp with id as the class index of each point
            gp_idx = class_idxs[idx]

            d, sigma = self.gps[gp_idx].predict(
                direction.reshape(1, -1), return_std=True
            )

            # compute the 3D point from the input phi,theta and the predicted
            # distance
            point_3D_gp = xyz_from_direction_distance_class_signle(
                direction, d, centers, gp_idx
            )

            # stack all the points and return them
            self.xyz.append(point_3D_gp)
            self.sigmas.append(sigma)
            self.dps.append(d)

        return (
            np.array(self.dps),
            np.array(self.xyz).reshape(-1, 3),
            np.array(self.sigmas),
        )

    def predict_(self, x, centers, center_idxs):
        xyz, sigmas, dps = [], [], []

        for i in np.unique(center_idxs):
            with CONSOLE.status(f"Infering distance...", spinner="bouncingBall"):
                time_s = time.time()
                dp, sigma = self.gps[i].predict(x[i], return_std=True)
                xyz.append(xyz_from_direction_distance_class(x[i], dp, centers, i))
                sigmas.append(sigma)
                dps.append(dp)
            CONSOLE.log(
                f"[bold green]:white_check_mark: Done infering with GP {i} with {len(dp)} assigned points."
            )
        self.xyz = np.concatenate(xyz, axis=0)
        self.sigmas = np.concatenate(sigmas, axis=0)
        self.dps = np.concatenate(dps, axis=0)

        return self.dps, self.xyz, self.sigmas

    def eval(self, original_points, pred_points):

        cd = chamfer_distance(pred_points, original_points)
        pd_mean,pd_med,pd_std = mean_p2p_distance(pred_points,original_points)

        table = Table(title="Eval metrics")
        table.add_column("CD", style="magenta")
        table.add_column("Mean pd", style="magenta")
        table.add_column("Median pd", style="magenta")
        table.add_column("Std pd", style="magenta")
        table.add_row(str(cd),str(pd_mean),str(pd_med),str(pd_std))

        CONSOLE.print(table)

        return cd

    def to3D(self, path, threshold=None):
        export_3D_points(self.xyz, os.path.join(path, "points_out.txt"))
