import numpy as np
from plyfile import PlyData,PlyElement
from numpy.typing import NDArray
import pymeshlab as ml

def fitModel2UnitSphere_(model,buffer):
    
    # find the centroid coordinates
    xmin,xmax = model[:,0].min(), model[:,0].max()
    ymin,ymax = model[:,1].min(), model[:,1].max()
    zmin,zmax = model[:,2].min(), model[:,2].max()
    
    # center
    center = np.array([xmax + xmin,ymax + ymin, zmax + zmin]) / 2
    
    # trasform vertices 
    model[:,0] -= center[0]
    model[:,1] -= center[1]
    model[:,2] -= center[2]
    
    # calculate max distance 
    max_distance = np.linalg.norm(model.max(axis=0)) * buffer
    
    model /= max_distance
    
    return model

def fitModel2UnitSphere(model,buffer):
    
    # find the centroid coordinates
    xmin,xmax = model[:,0].min(), model[:,0].max()
    ymin,ymax = model[:,1].min(), model[:,1].max()
    zmin,zmax = model[:,2].min(), model[:,2].max()
    
    # center
    center = np.array([xmax + xmin,ymax + ymin, zmax + zmin]) / 2
    
    # trasform vertices 
    model[:,0] -= center[0]
    model[:,1] -= center[1]
    model[:,2] -= center[2]
    
    # calculate max distance 
    distances = np.linalg.norm(model,axis=1) * buffer
    
    model /= distances.max()
    
    return model
def export_3D_points(points,filename):
    with open(filename,'w') as f:
        for i in points:
            f.write(f"{i[0]},{i[1]},{i[2]}\n")
def load3DModel(modelpath: str) -> NDArray:
    """Loads a 3D model of format .ply as an (n,3) array of (n,x,y,z).
    The resulting x,y,z are the vertices of the model.

    Args:
        modelpath (Literal): Path to disc of the 3D model.

    Returns:
        NDArray: Vertices of the model in a (N,x,y,z) array.
    """
    data = PlyData.read(modelpath)

    x = np.array(data['vertex']['x'])
    y = np.array(data['vertex']['y'])
    z = np.array(data['vertex']['z'])
    points = np.column_stack((x, y, z))

    return points
def clustering_decimation(input_mesh_path,output_mesh_path):
    ms = ml.MeshSet()
    threshold = ml.threshold(1)
    # load the input mesh
    ms.load_new_mesh(input_mesh_path)
    
    # apply the filter
    ms.apply_filter('meshing_decimation_clustering',threshold=threshold)
    #ms.meshing_decimation_clustering(0.01)
    # Save the result
    ms.save_current_mesh(output_mesh_path)
def writePLY(vertices,file):
    
    vertex = np.array(vertices,dtype=[('x','f'),('y','f'),('z','f')])
    el = PlyElement.describe(vertex,'vertex')
    PlyData([el]).write(file)
    
def spherical_coordinates_to_cartesian(phi, theta, d):
    x = d * np.sin(theta) * np.cos(phi)
    y = d * np.sin(theta) * np.sin(phi)
    z = d * np.cos(theta)
    return x,y,z
    