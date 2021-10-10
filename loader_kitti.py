import numpy as np
import os
import time
from loader import Loader


class LoaderKITTI(Loader):
    def __init__(self, dataset_path, sequence):
        self.N_SCANS = 64
        # self.folder_path = os.path.join(dataset_path, 'dataset', 'sequences', sequence, 'velodyne')  # original
        self.folder_path = os.path.join(dataset_path, sequence, 'velodyne')  # local folder structure
        self.pcds_list = os.listdir(self.folder_path)
        self.pcds_list.sort()

    def length(self):
        return len(self.pcds_list)

    def get_item(self, ind):
        path = os.path.join(self.folder_path, self.pcds_list[ind])
        pcd = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]  # 没有拿反射率
        return self.reorder_pcd(pcd)

    # vlp-64e [-24.8, 2], 360, [0.9,120]
    # scan_ids = ring number
    # 基于球形投影重新近似计算每个点的scan，存在一定误差，会忽略
    def _get_scan_ids(self, pcd):
        '''
        #old version 0.005s arctan vs. arcsin 0.0021s
        #t1=time.time()
        depth = np.linalg.norm(pcd[:, :3], 2, axis=1)  # x,y,z (n*3) 的2-norm, distance
        pitch = np.arcsin(pcd[:, 2] / depth)  # 垂直角分辨率
        '''
        #t1=time.time()
        pitch = np.arctan(np.divide(pcd[:, 2], np.sqrt(pcd[:, 0]**2 + pcd[:, 1]**2)))
        fov_down = -24.8 / 180.0 * np.pi  # vlp-64e v-fov [-24.8, 2]
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        scan_ids = (pitch + abs(fov_down)) / fov
        scan_ids *= self.N_SCANS
        scan_ids = np.floor(scan_ids)
        scan_ids = np.minimum(self.N_SCANS - 1, scan_ids)
        scan_ids = np.maximum(0, scan_ids).astype(np.int32)
        #t2=time.time()-t1
        return scan_ids
