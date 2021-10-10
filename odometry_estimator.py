#import mrob
import numpy as np
import open3d as o3d

from feature_extractor import FeatureExtractor
#from optimizer import LOAMOptimizer
from utils import get_pcd_from_numpy, matrix_dot_product
import utils
'''
整个代码特征命名方式过于混乱，sharp/edge planer/flat一直在交替使用
实际上是等价的，需要进行统一
'''
class OdometryEstimator:
    '''
    可能：DISTANCE_SQ_THRESHOLD的设置对heading 旋转速度较大的情况十分敏感，也就是当绕z轴旋转速度较快时，该阈值可能造成丢失大量
    可以考虑利用z轴角速度积分来限定距离阈值或其它策略
    '''
    DISTANCE_SQ_THRESHOLD = 1  # kitti_vlp64的0.1fps，1m约束近似10m/s=36km/h（数据采集车速基本<40km/h）, 为低速状态下的估计;自适应速度可由acc积分估算替换
    SCAN_VICINITY = 2.5  # VFOV=26.9 可认为是为了适配多线束、垂直分辨率小抖动匹配的; vlp16在适配时，垂直分辨率大不知道是否需要调小

    def __init__(self):
        self.extractor = FeatureExtractor()  # 特征提取器

        self.inited = False
        self.last_less_sharp_points = None  # 上一帧边特征
        self.last_less_flat_points = None   # 上一帧面特征
        self.last_position = np.eye(4)  # 位姿矩阵 用于每次初始化配准

    def append_pcd(self, pcd):
        # input: 点序列 起点索引 终点索引
        # output: 强边、边、强面、面
        sharp_points, less_sharp_points, flat_points, less_flat_points = self.extractor.extract_features(pcd[0], pcd[1], pcd[2])
        pcd=utils.get_pcd_from_numpy(np.vstack(pcd[0]))
        pcd.paint_uniform_color([0.1,0.1,0.1])
        sharp = utils.get_pcd_from_numpy(np.vstack(sharp_points))
        sharp.paint_uniform_color([1, 0, 0])
        less_sharp= utils.get_pcd_from_numpy(np.vstack(less_sharp_points))
        less_sharp.paint_uniform_color([0, 1, 0])
        flat = utils.get_pcd_from_numpy(np.vstack(flat_points))
        flat.paint_uniform_color([0, 0, 1])
        less_flat= utils.get_pcd_from_numpy(np.vstack(less_flat_points))
        less_flat.paint_uniform_color([1, 1, 0])
        utils.pc_show([sharp,less_sharp,flat])
        #utils.pc_show([sharp,less_sharp,flat,less_flat])


    '''
    当前帧的 edge特征i 在上一帧的 less_edge找到相关的参考点集
    基于‘点线距离’求解准备，针对t时刻的特征，需要找到t-1时刻的两个相邻特征
    先找到距离i最近的一个特征点j，
    然后找到与j不在同一扫描线上的l，l到i最近，且在j的前后2根扫描线上。
    共同构成(j,l)
    '''
    def find_edge_correspondences(self, sharp_points):

        corners_cnt = len(sharp_points)

        edge_points = []
        edge_1 = []
        edge_2 = []
        # 基于上一帧点云less_edge特征构建KDtree
        less_sharp_points_tree = o3d.geometry.KDTreeFlann(get_pcd_from_numpy(self.last_less_sharp_points))
        # 遍历当前强边特征(edge[t])
        for i in range(corners_cnt):
            point_sel = sharp_points[i]  # 当前点(xyzr)
            _, idx, dist = less_sharp_points_tree.search_knn_vector_3d(point_sel[:3], 1)  # 数量默认1,不进行获取，返回kd-tree中最近的knn=1个邻居点
            min_point_ind_2 = -1
            if dist[0] < self.DISTANCE_SQ_THRESHOLD:  # 在距离1m范围内
                closest_point_ind = idx[0]  # j点
                min_point_sq_dist_2 = self.DISTANCE_SQ_THRESHOLD
                closest_point_scan_id = self.last_less_sharp_points[closest_point_ind][3]  # j点的ring number
                # 当前点到所有 边特征的“距离”
                dist_to_sel_point = matrix_dot_product((self.last_less_sharp_points[:, :3] - point_sel[:3]),
                                                       (self.last_less_sharp_points[:, :3] - point_sel[:3]))
                # 从排除j之后，开始遍历上一帧点云 less sharp特征的剩余点寻找l
                # 上扫描线遍历[r + 1, r + 2]
                for j in range(closest_point_ind + 1, len(self.last_less_sharp_points)):
                    #  只有扫描线不超过 对应点scan number的情况下进行，需要排除当前扫描线上的点
                    if self.last_less_sharp_points[j][3] <= closest_point_scan_id:
                        continue
                    if self.last_less_sharp_points[j][3] > closest_point_scan_id + self.SCAN_VICINITY:  #
                        break
                    # 满足扫描线约束的点距离
                    point_sq_dist = dist_to_sel_point[j]
                    if point_sq_dist < min_point_sq_dist_2:  # 到他距离小于1的邻居点
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j
                # 下扫描线遍历 [r - 1, r - 2]
                for j in range(closest_point_ind - 1, -1, -1):
                    if self.last_less_sharp_points[j][3] >= closest_point_scan_id:  # >=ring
                        continue
                    if self.last_less_sharp_points[j][3] < closest_point_scan_id - self.SCAN_VICINITY:  # <ring-2.5
                        break

                    point_sq_dist = dist_to_sel_point[j]
                    if point_sq_dist < min_point_sq_dist_2:
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j
                #  添加寻在的特征点对 t:t-1
                if min_point_ind_2 >= 0:
                    edge_points.append(point_sel)  # E(t)特征点
                    edge_1.append(self.last_less_sharp_points[closest_point_ind])  # 对应j
                    edge_2.append(self.last_less_sharp_points[min_point_ind_2])  # 对应l
        #  入队
        edge_points = np.vstack(edge_points)[:, :3]
        edge_1 = np.vstack(edge_1)[:, :3]
        edge_2 = np.vstack(edge_2)[:, :3]

        return edge_points, edge_1, edge_2
    '''
    基于当前点F(t)中的i， 找到F(t-1)中的 不共线的三点j l m
    基于 点到平面距离，构建correspondence
    先找到距离i最近的一点j
    然后找另外两个到i最近的点l，m
        其中，l与j共扫描线，但l!=j
        其中，m在j的上下扫描线上/scan+-2, zhangji17仅考虑上下1根相邻扫描线
    '''
    def find_surface_correspondences(self, flat_points, pcd):  # pcd for visualization only
        surface_cnt = len(flat_points)
        print('Surface count: ', surface_cnt)

        surface_points = []  # i \in flap[t]
        surface_1 = []  # j \in less_flap[t-1]
        surface_2 = []  # m \in less_flap[t-1]
        surface_3 = []  # l \in less_flap[t-1]

        less_flat_points_tree = o3d.geometry.KDTreeFlann(get_pcd_from_numpy(self.last_less_flat_points))  # F(t-1)的KD-tree
        for i in range(surface_cnt):
            point_sel = flat_points[i]  # 点I
            _, idx, dist = less_flat_points_tree.search_knn_vector_3d(point_sel[:3], 1)  # 搜索到I最近的点
            min_point_ind_2 = -1
            min_point_ind_3 = -1
            # less_flat[t-1]中所有点到i的距离
            dist_to_sel_point = matrix_dot_product((self.last_less_flat_points[:, :3] - point_sel[:3]),
                                                   (self.last_less_flat_points[:, :3] - point_sel[:3]))

            closest_point_ind = idx[0]  # 点J
            v = self.last_less_flat_points[closest_point_ind][:3] - point_sel[:3]
            dist = np.dot(v, v)  # J到I的‘距离’, 满足1m内约束条件
            if dist < self.DISTANCE_SQ_THRESHOLD:
                closest_point_scan_id = self.last_less_flat_points[closest_point_ind][3]  # J的scan id
                min_point_sq_dist_2 = self.DISTANCE_SQ_THRESHOLD  # 1m
                min_point_sq_dist_3 = self.DISTANCE_SQ_THRESHOLD  # 1m

                for j in range(closest_point_ind + 1, len(self.last_less_flat_points)):
                    if self.last_less_flat_points[j][3] > closest_point_scan_id + self.SCAN_VICINITY:  # [scan_j+3,63]
                        break

                    point_sq_dist = dist_to_sel_point[j]
                    # [0, scan_j]
                    if self.last_less_flat_points[j][3] <= closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_2:  # l与j共扫描线, 其余靠距离阈值过滤
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j
                    # [scan_j+1, scan_j+2]
                    elif self.last_less_flat_points[j][3] > closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_3:  # m与j不共扫描线
                        min_point_sq_dist_3 = point_sq_dist
                        min_point_ind_3 = j

                for j in range(closest_point_ind - 1, -1, -1):
                    if self.last_less_flat_points[j][3] < closest_point_scan_id - self.SCAN_VICINITY:  # [0,scan_j-3]
                        break

                    point_sq_dist = dist_to_sel_point[j]
                    # [scan_j,63]
                    if self.last_less_flat_points[j][3] >= closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_2:
                        min_point_sq_dist_2 = point_sq_dist
                        min_point_ind_2 = j
                    # [scan_j-2,scan_j-1]
                    elif self.last_less_flat_points[j][3] < closest_point_scan_id \
                            and point_sq_dist < min_point_sq_dist_3:
                        min_point_sq_dist_3 = point_sq_dist
                        min_point_ind_3 = j

                if min_point_ind_2 >= 0 and min_point_ind_3 >= 0:
                    surface_points.append(point_sel)  # from current pcd[t]
                    # from last frames
                    surface_1.append(self.last_less_flat_points[closest_point_ind])  # 点J
                    surface_2.append(self.last_less_flat_points[min_point_ind_2])  # 点L
                    surface_3.append(self.last_less_flat_points[min_point_ind_3])  # 点 M

        surface_points = np.vstack(surface_points)  # i
        surface_1 = np.vstack(surface_1)  # j
        surface_2 = np.vstack(surface_2)  # l
        surface_3 = np.vstack(surface_3)  # m
        ind = surface_1[:, 3] > 0  # 逻辑上应该所有 scan id都大于0, 排除最近一圈在车上的？

        ''' i的对应特征点jlm，应该越多越好，而且应该是满足距离阈值（1m）越小越好
        # 可视化将前后两帧的特征点投影在同一坐标系下，满足相对T计算前提
        print('output: ', surface_points.shape[0])
        surf = np.vstack((surface_1[ind], surface_2[ind], surface_3[ind]))  # t-1 中的参考点 j+l+m
        keypoints = utils.get_pcd_from_numpy(surf)
        keypoints.paint_uniform_color([0, 1, 0])  # green
        pcd = utils.get_pcd_from_numpy(pcd[0])  # t时刻面特征（存在对应参考点的）
        pcd.paint_uniform_color([0, 0, 1])  # blue
        orig = utils.get_pcd_from_numpy(surface_points[ind])  # 使用t-1的索引调t时刻pcd中的点
        orig.paint_uniform_color([1, 0, 0])  # red
        # o3d.visualization.draw_geometries([pcd, keypoints, orig])  # t, t-1, t| _:3:1
        o3d.visualization.draw_geometries([keypoints, orig])
        '''
        return surface_points[ind][:, :3], surface_1[ind][:, :3], surface_2[ind][:, :3], surface_3[ind][:, :3]
