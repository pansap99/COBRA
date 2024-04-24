import matplotlib.pyplot as plt
import vtk
import sys
from utils.gp_utils import *
import numpy as np
from matplotlib import cm
import open3d as o3d
from pose_vis.camera import PerspectiveCamera
from OpenGL.GL import *
import cv2 as cv
from os.path import join as jn

default_matrix = vtk.vtkMatrix4x4()
def boundingBoxPerInstance(rot,tr,obj_points,K,obj_id):
    """Computes the bounding box of each instance present in the image
    by projecting some known 3D points of the model to the image and finding 
    the Lower-Left and Upper-Right points of the boudning box by finding the 
    maximum and minimum of the projected points in image coordinates.

    Args:
        rot (ND ARRAY): Rotation matrix 3 x 3
        tr (ND ARRAY): Translation vector 1 x 3
        obj_points (ND ARRAY): The loaded known 3D points of the model
        K (ND ARRAY): The calibration matrix
        obj_id (int): The object's id

    Returns:
        tuple: Lower-Left,Upper-Right points of the computed bounding box
    """

    result,_= cv.projectPoints(obj_points,
                              rot,
                              tr,
                              cameraMatrix=K,
                              distCoeffs=None)
   
    # calculate the lower-left and upper-right of the bounding bot
    LL = (result[:,...,0].min() , result[:,...,1].max())
    UR = (result[:,...,0].max() , result[:,...,1].min())

    return LL,UR
def renderPose(vertices,
               indices,
               renderer,
               objID,
               conf,
               threshold,
               resolution,
               RT,
               K,
               savePath,
               mesh_color= [1.0,0.5,0.31],
               rgb_image = None):
    

    camera = PerspectiveCamera(resolution[0],resolution[1])
    projection = camera.fromIntrinsics(
        fx = K[0,0],
        fy = K[1,1],
        cx = K[0,2],
        cy = K[1,2],
        nearP = 1,
        farP=5000
    )

    model_ = np.eye(4)

    # configure rendering params
    uniform_vars = {"objectColor": {"value":mesh_color,"type":'glUniform3fv'}, # 1.0, 0.5, 0.31
                    "lightColor":{"value": [1.0, 1.0, 1.0],"type":'glUniform3fv'},
                    "lightPos":{"value": [0.0, 0.0 , 0.0],"type":'glUniform3fv'},
                    "viewPos":{"value": [0.0, 0.0, 0.0],"type":'glUniform3fv'},
                    "model":{"value": model_,"type":'glUniformMatrix4fv'},
                    "view":{"value":RT,"type":'glUniformMatrix4fv'},
                    "projection":{"value": projection,"type":'glUniformMatrix4fv'},
                    }
    LL,UR = boundingBoxPerInstance(RT[:3,:3],RT[:3,-1],vertices,K.reshape(3,3),objID)
    UL,LR = (LL[0],UR[1]),(UR[0],LL[1])


    RT = renderer.cv2gl(RT)

    # adjust lighting position
    lightPos = np.dot(np.array([RT[0,-1],RT[1,-1],RT[2,-1],1.0]),
                      np.linalg.inv(RT))
    
    # update uniform variables
    uniform_vars["view"]["value"] = RT
    uniform_vars["lightPos"]["value"] = [lightPos[0],lightPos[1],lightPos[2]]
    uniform_vars["viewPos"]["value"] = [-RT[0,-1], -RT[1,-1], -RT[2,-1]]

    renderer.setUniformVariables(renderer.shader_programm,uniform_vars)
    glBindVertexArray(renderer.VAO)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    renderer.ProjectFramebuffer(renderer.framebuffer,resolution)

    if not rgb_image:
        mask = renderer.CaptureFramebufferScene(jn(savePath),saveRendered=True)
    else:
        mask = renderer.CaptureFramebufferScene(jn(savePath,'test.png'),saveRendered=False)
        renderer.draw2DBoundingBox(cv.imread(rgb_image).astype(np.float32),
                                   mask.astype(np.float32),
                                   str(objID),
                                   conf=conf,
                                   savePath=savePath,
                                   bb=np.array([UL,LR]).astype(int),
                                   threshold=threshold,
                                   buildMask=False,
                                   maskFolder=None,
                                   opacity=0.6
                                   )
        
def vis_pcd_open3D(points,savePath):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points =  o3d.utility.Vector3dVector(points)
    #pcd.estimate_normals()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_zoom(1)
    
    # Run the visualizer
    vis.run()
    vis.capture_screen_image(savePath)
    
    vis.destroy_window()
    
def vis_points_o3d(points, sphere_radius=0.05, model_3d=None,center_points=None):
    # Create a point cloud for the original points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0,0,1.0])  # Set original points color to white

    # Create a sphere for each center point
    if center_points is not None:
        center_spheres = []
        for center_point in center_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.paint_uniform_color([1, 0, 0])  # Set center points color to blue
            sphere.translate(center_point)
            center_spheres.append(sphere)

    # Visualize the combined point cloud and spheres
    if model_3d is not None:
        mesh = o3d.io.read_triangle_mesh(model_3d)

        if center_points is not None:
            o3d.visualization.draw_geometries([pcd, *center_spheres, mesh], point_show_normal=False, mesh_show_back_face=True,width=800, height=800)
        else:
            o3d.visualization.draw_geometries([pcd, mesh], point_show_normal=False, mesh_show_back_face=True,width=800, height=800)
    else:
        if center_points is not None:
            o3d.visualization.draw_geometries([pcd, *center_spheres], point_show_normal=False,width=800, height=800)
        else:
            o3d.visualization.draw_geometries([pcd], point_show_normal=False,width=800, height=800)
    
def create_colored_points_renderer(points, class_indices,ppclass,duplicates,colormap='Dark2'):
    
    class_indices = sort_class_indices(class_indices,ppclass)
    # Create a VTK renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderer.SetBackground([1,1,1])
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800,800)
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Create a vtkPoints object and add the points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)

    # Create a vtkCellArray to represent the points
    vertices = vtk.vtkCellArray()
    for i in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)

    # Create a vtkPolyData object and set the points and vertices
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetVerts(vertices)

   # Create a vtkLookupTable for coloring based on class indices
    num_classes = len(set(class_indices))
    color_map = vtk.vtkLookupTable()
    color_map.SetNumberOfTableValues(num_classes)
    color_map.Build()

   # Create a vtkUnsignedCharArray to store the colors
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    cmap = cm.get_cmap(colormap)
    cmp = cmap(np.linspace(0, 1, len(ppclass)))
    
    for i, class_index in enumerate(class_indices):
        
        if duplicates is not None:
            if not duplicates[i]:
                color_map.SetTableValue(class_index, cmp[class_index][0] , cmp[class_index][1] , cmp[class_index][2] , 1.0)
                colors.InsertNextTuple3(int(cmp[class_index][0]*255), int(cmp[class_index][1]*255), int(cmp[class_index][2]*255))
            else:
                color_map.SetTableValue(class_index, 0 , 0 ,1 , 1.0)
                colors.InsertNextTuple3(int(0*255), int(0*255), int(1*255))
        else:
            color_map.SetTableValue(class_index, cmp[class_index][0] , cmp[class_index][1] , cmp[class_index][2] , 1.0)
            colors.InsertNextTuple3(int(cmp[class_index][0]*255), int(cmp[class_index][1]*255), int(cmp[class_index][2]*255))
    # Set the colors as point data in the polydata
    polydata.GetPointData().SetScalars(colors)

    # Create a vtkPolyDataMapper and a vtkActor
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()

    mapper.SetInputData(polydata)
    mapper.SetScalarRange(min(class_indices), max(class_indices))
    mapper.SetLookupTable(color_map)

    actor.SetMapper(mapper)

    # Add the actor to the renderer
    # Set point size in the renderer
    actor.GetProperty().SetPointSize(5)
    renderer.AddActor(actor)

    renderer.ResetCamera()
    render_window.Render()
    render_window_interactor.Start()
    
    return renderer, render_window, render_window_interactor
