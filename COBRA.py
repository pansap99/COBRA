import sys

sys.path.append("..")
from utils.gp_utils import *
import numpy as np
from numpy.typing import *
from sklearn.cluster import KMeans


def transform_2D_to_3D(K, exrtinsics, P2D, P3D):

    # Invert the calibration matrix
    K_inv = np.linalg.inv(K)

    # Extract rotation matrix R and translation vector t from the extrinsic matrix
    R = exrtinsics[:, :3]
    t = exrtinsics[:, 3]

    # compute lambda
    l = ((K_inv @ P2D).T @ (R @ P3D + t)) / np.linalg.norm(K_inv @ P2D, ord=2) ** 2
    P3D_est = l * R.T @ K_inv @ P2D - R.T @ t

    return P3D_est


def transform_3D_to_2D(K, extrinsics, P3D):

    p2d_homo = K @ extrinsics[:3, :] @ P3D
    return p2d_homo[:-1] / p2d_homo[-1]


def compute_likelihood(point_3D_ref, p3d_observed, sigma=1.0, weight=1):

    eu_distance = np.linalg.norm(point_3D_ref - p3d_observed)
    likelihood = weight * mt.exp(-0.5 * (eu_distance**2) / (sigma**2))
    #likelihood = 1/(mt.sqrt(2*mt.pi) * sigma) * weight * mt.exp(-0.5 * (eu_distance**2) / (sigma**2))

    return eu_distance, likelihood


class COBRA:
    def __init__(self, gps: ClusteredGPs, path) -> None:
        gps.__load__(path)
        self.gps = gps
        self.centers = gps.centers

    def calc_conf_lower_bound(self, delta, std_template, weights):
       

        # norm_factor = (1/ (delta**2)) * std_template
        # wsum = std_template * (1 - mt.exp(-((delta**2)/(2*std_template))))

        # return norm_factor * in_sum

        norm_factor = 1 / (mt.sqrt(2*mt.pi) * delta**2)
        wsum = np.sum(
            weights
            * std_template
            * (1 - mt.exp(-((delta**2) / (2 * std_template**2)))),
            axis=0,
        )
        return norm_factor * wsum

    def score_pose(
        self,
        points2D: NDArray,
        points3D: NDArray,
        RT: NDArray,
        K: NDArray,
        sigma_hat: float,
        weights: NDArray = None,
        delta: float = 5,
    ):

        back_proj_3D = []
        # back-project 2D points using the estimate pose
        # compute the 3D point across the ray with least squares
        for p2d, p3d in zip(points2D, points3D):
            back_proj_3D.append(transform_2D_to_3D(K, RT, np.append(p2d, 1.0), p3d))

        back_proj_3D = np.asarray(back_proj_3D)

        # classify backprojected points to reference points
        _, class_idxs = classify_points_to_center(back_proj_3D, self.gps.centers)

        # phi,theta,d parameterization
        distances = distance_from_centers(back_proj_3D, self.gps.centers, class_idxs)
        _, phi_thetas, ds_observed, sorted_indices = direction_distance_given_class(
            back_proj_3D, distances, self.gps.centers, class_idxs
        )
        # predict d given phis,thetas and compute the 3D point location
        # in the coordinate frame of the object
        _, xyz, _ = self.gps.predict_(phi_thetas, self.centers, class_idxs)

        likelihoods = []
        distances = []

        # if weights are not provided default to 1/N
        if weights is None:
            weights = np.ones((len(xyz))) * (1/len(xyz))
        # compute the likelihood for each point
        for idx, (xyz_, xyz_ob) in enumerate(zip(xyz, back_proj_3D[sorted_indices])):
            
            eu_distance, likelihood = compute_likelihood(
                xyz_, xyz_ob, sigma=sigma_hat, weight=weights[idx]
            )

            likelihoods.append(likelihood)
            distances.append(eu_distance)

        # compute lower bound for confidence score
        conf_lower_bound = self.calc_conf_lower_bound(
            delta=delta, std_template=sigma_hat, weights=weights
        )
        print(np.sum(weights))
        return (
            np.sum(np.array(likelihoods)) / np.sum(weights),
            np.array(distances),
            np.array(conf_lower_bound) / np.sum(weights),
            xyz
        )
