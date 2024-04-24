import sys
import numpy as np
import pickle
from absl import app,flags
from COBRA import COBRA
sys.path.append("..")
from utils.gp_utils import ClusteredGPs
from utils.io import loadEstimatorResults
from vis.vis import *
import glob
import json
from pose_vis.Renderer import Renderer
from pose_vis.utils import load_model
from os.path import join as jn
from utils.common import *

        
FLAGS = flags.FLAGS

flags.DEFINE_float('sigma_hat',None,'Sigma hat derived from eval')
flags.DEFINE_string('model_3d',None,'3D model inside the common.MODEL_PATH_ORIGINAL')
flags.DEFINE_string('centers',None,'Centers txt file')
flags.DEFINE_integer('objID',1,'Object id to visualize')
flags.DEFINE_float('delta',2,'Delta value to compute lower confidence bound')

def main(args):

    # load camera intrinsics
    K = np.loadtxt(jn(SCORING_PATH,FLAGS.model_3d,'K.txt'),delimiter=',')

    # if the centers flag is not provided load the kmeans
    if FLAGS.centers is None:
        # load kmeans
        with open(jn(RESULTS_PATH,FLAGS.model_3d,'kmeans.pkl'),'rb') as f:
            kmeans = pickle.load(f)
        centers = kmeans.cluster_centers_
    else:
        centers = np.loadtxt(FLAGS.kmeans,delimiter=',')

    # load ground-truth and estimated poses from json file
    with open(jn(SCORING_PATH,FLAGS.model_3d,'est_poses.json'),'r') as f:
        poses = json.load(f)

    # instantiate a ClusteredGPs object
    gps = ClusteredGPs(centers)

    # initialize COBRA
    cobra = COBRA(gps,jn(RESULTS_PATH,FLAGS.model_3d))
    
    # initialize renderer
    renderer = Renderer(bufferSize=(640,480))
    renderer.load_shaders("./vis/shaders/basic_lighting_vrt.txt",
                        "./vis/shaders/basic_lighting.txt",
                        None)
    vertices, indices = load_model(jn(MODEL_PATH,'original',FLAGS.model_3d)+'.ply')
    renderer.create_data_buffers(vertices,indices,attrs=[2,3,4])
    renderer.CreateFramebuffer(GL_RGB32F,GL_RGBA,GL_FLOAT)


    # get all corr files present in the corrs folder
    corrs = sorted(glob.glob(jn(SCORING_PATH,'corr')+"/*corr"))
    
    data = {}
    distances_out = []
    likelihoods_ = []
    distances_ = []
   
    for img in glob.glob(jn(SCORING_PATH,FLAGS.model_3d,'images') + "/*.png"):
        
        renderer.glConfig()
        corr = jn(SCORING_PATH,FLAGS.model_3d,'corrs',str(int(os.path.basename(img).split(".")[0]))+"_corr.txt")
        
        # load the corr file into a pandas dataframe
        corr_df = loadEstimatorResults(corr)
    
        # get the corresponding pose
        pose_id = str(int(os.path.basename(corr).split('_')[0]))
        RT = np.eye(4)
        R = np.array(poses[pose_id][0]['cam_R_m2c']).reshape(3,3)
        T = np.array(poses[pose_id][0]['cam_t_m2c'])
        RT[:3,:3] = R
        RT[:3,-1] = T

        # get only the inliers
        #inliers = corr_df[corr_df.iloc[:,1].to_numpy().astype(int) ==1]
        
        # extract the 2D and 3D points
        points_3D = np.column_stack((corr_df['X'],corr_df['Y'],corr_df['Z']))
        points_2D = np.column_stack((corr_df['x'],corr_df['y']))
        if 'conf' in corr_df.columns:
            estimator_weights = corr_df['conf']
        else:
            estimator_weights = None

        # compute COBRA confidence score
        likelihood,distances,conf_lower_bound,xyz = cobra.score_pose(points_2D,
                                                                points_3D,
                                                                RT[:3,:],
                                                                K,
                                                                FLAGS.sigma_hat,
                                                                None,
                                                                delta=FLAGS.delta
                                                                )
        # visualize and save results
        renderPose(vertices.reshape(-1,3),
                indices,
                renderer,
                objID=FLAGS.objID,
                conf=likelihood,
                threshold=conf_lower_bound,
                resolution=(640,480),
                RT= RT,
                K = K.reshape(3,3),
                savePath= jn(SCORING_PATH,FLAGS.model_3d,'vis',pose_id) + ".png",
                mesh_color=[1.0, 0.5, 0.31],
                rgb_image=img
                )
        
        print(f"Image ID: {pose_id}, confidence: {likelihood}, distances mean: {distances.mean()}")
    print("CONF_LOWER_BOUND: ",conf_lower_bound)
if __name__ == "__main__":
    app.run(main)