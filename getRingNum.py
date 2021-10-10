import numpy as np
import open3d as o3d
from loader_kitti import LoaderKITTI
import utils
from odometry_estimator import OdometryEstimator


if __name__ == '__main__':
    folder= './data/'
    loader = LoaderKITTI(folder, '00')
    odometry = OdometryEstimator()
    for i in range(loader.length()):
        pcd = loader.get_item(i)
        #utils.pc_show([pcd[0]])  # 原始点云
        #utils.show_ring_num(pcd)  # 基于scan_id着色
        odometry.append_pcd(pcd)  # 显示不同特征

