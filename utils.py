import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt # for colormap
# array to o3d.pcd
def get_pcd_from_numpy(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    pcd.paint_uniform_color([0,0,0])
    return pcd

def pc_show(pc,norm_flag=False):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800,height=800)
    opt = vis.get_render_option()
    opt.point_size = 2
    opt.point_show_normal=norm_flag
    for p in pc:
        if type(p)!=o3d.cpu.pybind.geometry.PointCloud:
            p=get_pcd_from_numpy(p)
        vis.add_geometry(p)
    vis.run()
    vis.destroy_window()

def matrix_dot_product(A, B):
    """
    Fast way to calculate dot product of every row in two matrices A and B
    Returns array like:
    [ np.dot(A[0], B[0]),
    np.dot(A[1], B[1]),
    np.dot(A[2], B[2]),
    ... ,
    np.dot(A[M - 1], B[M - 1])]

    :param A: MxN numpy array
    :param B: MXN numpy array
    :return: Mx1 numpy array
    """
    assert A.shape == B.shape
    return np.einsum('ij,ij->i', A, B)  # A_ij B_ij

def show_ring_num(pcd):
    points = pcd[0]  #
    ring = points[:, 3] / max(points[:, 3])  # unify intensity
    pcd = o3d.geometry.PointCloud()
    print(type(pcd))
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('jet')(ring)[:, 0:3])  # RGBA2RGB
    pc_show([pcd])