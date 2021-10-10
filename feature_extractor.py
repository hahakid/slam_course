#import math
import numpy as np
import open3d as o3d
from utils import matrix_dot_product
from utils import pc_show
#import utils
'''
当前函数中存在大量基于 相邻点坐标做差，然后求向量点乘，作为判断的阈值条件： 0.05, 0.2， 0.3。
基于余弦定理，其实代表了两个激光点的 夹角 或者 空间距离的平方
a·b>0 方向基本相同，夹角在0°到90°之间 <--
a·b=0 正交，相互垂直  
a·b<0 方向基本相反，夹角在90°到180°之间 
'''
class FeatureExtractor:
    # Number of segments to split every scan for feature detection
    N_SEGMENTS = 6

    # Number of less sharp points to pick from point cloud
    PICKED_NUM_LESS_SHARP = 20
    # Number of sharp points to pick from point cloud
    PICKED_NUM_SHARP = 4
    # Number of less sharp points to pick from point cloud
    PICKED_NUM_FLAT = 4
    # Threshold to split sharp and flat points
    SURFACE_CURVATURE_THRESHOLD = 0.1
    # Radius of points for curvature analysis (S / 2 from original paper, 5A section)
    FEATURES_REGION = 5  # window length (10+1) for sharpness, 当前点的前后5个点

    # idx为0-1序列flag,0-不选，1-选取，可视化使用
    def selectedPC(self, pc, idx):
        # 基于idx flag生成list
        count=0
        ind=[]
        for i in idx:
            count+=i
            if i != 0:
                ind.append(int(count))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
        sel_pcd=pcd.select_by_index(ind)
        sel_pcd.paint_uniform_color([1.0,0,0])
        pcd.paint_uniform_color([0, 1, 0])
        #pc_show([sel_pcd])
        pc_show([pcd,sel_pcd])

    def extract_features(self, laser_cloud, scan_start, scan_end):
        keypoints_sharp = []
        keypoints_less_sharp = []
        keypoints_flat = []
        keypoints_less_flat = []

        cloud_curvatures = self.get_curvatures(laser_cloud)  # 基于前后点，获取每个点平滑度/曲率

        cloud_label = np.zeros((laser_cloud.shape[0]))  # empty labels for storing，初始化为0
        cloud_neighbors_picked = np.zeros((laser_cloud.shape[0]))
        # 剔除不可靠点，在遍历计算平滑度之后，拆分特征点之前。
        cloud_neighbors_picked = self.remove_unreliable(cloud_neighbors_picked, laser_cloud, scan_start, scan_end)
        #print("outliner:",sum(cloud_neighbors_picked))
        # i=扫描线[0-63]; j=区块[0-5]; k=平滑度特征选取窗口[]
        for i in range(scan_end.shape[0]):  # 1.基于扫描线遍历
            s = scan_start[i] + self.FEATURES_REGION  # 预留出前5
            e = scan_end[i] - self.FEATURES_REGION - 1  # 提前结束，保障窗口最后还有5
            if e - s < self.N_SEGMENTS:  # 越界保护
                continue
            # 0-5 一个ring拆分6等份操作， 2.基于子区间遍历
            for j in range(self.N_SEGMENTS):
                sp = s + (e - s) * j // self.N_SEGMENTS
                ep = s + (e - s) * (j + 1) // self.N_SEGMENTS - 1
                segments_curvatures = cloud_curvatures[sp:ep + 1]  # 当前子环子段的连续11个平滑度值
                sort_indices = np.argsort(segments_curvatures)  # 基于平滑度递增排序，平面在前 边在后
                '''
                边特征筛选，先反向排序， 非平滑值在末端
                '''
                largest_picked_num = 0
                for k in reversed(range(ep - sp)):
                    # print(ep,sp)
                    if i < 45:  # 边特征的选取范围仅考虑 45-64的扫描线？ 针对kitti数出来不计算地面的点。
                        break
                    ind = sort_indices[k] + sp  # 在当前点云帧的全局序号索引
                    # 不在之前选取的点范围内 and 平滑度大于0.5 (论文是0.005,其它改版为0.1，需测试) and edge判断条件
                    if cloud_neighbors_picked[ind] == 0 and cloud_curvatures[ind] > 0.5 and \
                            self.can_be_edge(laser_cloud, ind):
                        largest_picked_num += 1
                        if largest_picked_num <= self.PICKED_NUM_SHARP:
                            keypoints_sharp.append(laser_cloud[ind])  # 添加边缘点
                            keypoints_less_sharp.append(laser_cloud[ind])  # 添加边缘点
                            cloud_label[ind] = 2
                        elif largest_picked_num <= self.PICKED_NUM_LESS_SHARP:  # keypoints_sharp 属于 keypoints_less_sharp
                            keypoints_less_sharp.append(laser_cloud[ind])  # 添加边缘点
                            cloud_label[ind] = 1
                        else:
                            break

                        cloud_neighbors_picked = self.mark_as_picked(laser_cloud, cloud_neighbors_picked, ind)  # 在已选择列表中标记
                        #print("outliner+edge:", sum(cloud_neighbors_picked))
                '''
                面特征筛选, 使用原递增排序排序
                '''
                smallest_picked_num = 0
                for k in range(ep - sp):
                    if i < 50:  # 边特征的选取范围仅考虑 50-64的扫描线？ 针对kitti数出在地面的扫描线,来不计算地面的点。
                        break
                    ind = sort_indices[k] + sp
                    # 不在之前选取的点范围内 and 平滑度小于0.1
                    if cloud_neighbors_picked[ind] == 0 and cloud_curvatures[ind] < self.SURFACE_CURVATURE_THRESHOLD:
                        smallest_picked_num += 1
                        cloud_label[ind] = -1
                        keypoints_flat.append(laser_cloud[ind])  # 添加平面点

                        if smallest_picked_num >= self.PICKED_NUM_FLAT:
                            break

                        cloud_neighbors_picked = self.mark_as_picked(laser_cloud, cloud_neighbors_picked, ind)
                        #print("outliner+edge+planer:", sum(cloud_neighbors_picked))
                # 增补面特征 less_flat, 没有限制上限(高达8w左右)， 主要集中在地面，因为没有过滤扫描线
                # 这部分特征点并没有被 mark_as_picked
                for k in range(sp, ep + 1):
                    # =0的点为未选取点(cloud_label初始默认全0) <0的点为平面点 and not 存在大间隙，前后遮挡物体，保证是一个确定的平面
                    if cloud_label[k] <= 0 and cloud_curvatures[k] < self.SURFACE_CURVATURE_THRESHOLD \
                            and not self.has_gap(laser_cloud, k):
                        keypoints_less_flat.append(laser_cloud[k])
        '''
        keypoints = utils.get_pcd_from_numpy(np.vstack(keypoints_flat))
        keypoints.paint_uniform_color([0, 1, 0])
        keypoints_2 = utils.get_pcd_from_numpy(np.vstack(keypoints_sharp))
        keypoints_2.paint_uniform_color([1, 0, 0])
        pcd = utils.get_pcd_from_numpy(laser_cloud)
        pcd.paint_uniform_color([0, 0, 1])
        #o3d.visualization.draw_geometries([keypoints])
        o3d.visualization.draw_geometries([keypoints, keypoints_2])
        '''
        return keypoints_sharp, keypoints_less_sharp, keypoints_flat, keypoints_less_flat
    '''
    c=1/|S||Xi| |Sum(Xi-Xj)|
    Xi与Xj差异越大，总的c越大，就是边特征 Edge/Sharp
    反之，特征处于平面，差异不大，属于面特征 planer
    '''
    def get_curvatures(self, pcd):
        coef = [1, 1, 1, 1, 1, -10, 1, 1, 1, 1, 1]
        #coef = [1, -10, 1]
        assert len(coef) == 2 * self.FEATURES_REGION + 1
        discr_diff = lambda x: np.convolve(x, coef, 'valid')  # 参数x，传入 f(x)=np.convolve() 进行计算
        x_diff = discr_diff(pcd[:, 0])
        #x_diff1 = discr_diff(pcd[:10, 0])
        y_diff = discr_diff(pcd[:, 1])
        z_diff = discr_diff(pcd[:, 2])
        curvatures = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
        #temp=pcd[self.FEATURES_REGION:-self.FEATURES_REGION]
        #temp1=np.linalg.norm(temp,axis=1)  # sqrt(x^2+y^2+z^2)
        curvatures /= np.linalg.norm(pcd[self.FEATURES_REGION:-self.FEATURES_REGION], axis=1) * 10  # *S
        curvatures = np.pad(curvatures, self.FEATURES_REGION)  # 前后0填充各5个数值，补齐到原pcd长度
        return curvatures

    # 在排除已选择的特征点的同时，还对其周围一定范围内的点进行排除，防止后续被选择
    # 但是这个排除还是基于扫描线的排除，而不是完整的空间距离
    def mark_as_picked(self, laser_cloud, cloud_neighbors_picked, ind):
        cloud_neighbors_picked[ind] = 1  # 当前点
        # ind 前后5个点序列 错位相减
        diff_all = laser_cloud[ind - self.FEATURES_REGION + 1:ind + self.FEATURES_REGION + 2] - \
                   laser_cloud[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION + 1]
        # 距离差矩阵点乘==向量夹角
        sq_dist = matrix_dot_product(diff_all[:, :3], diff_all[:, :3])
        # 1：6
        for i in range(1, self.FEATURES_REGION + 1):
            if sq_dist[i + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + i] = 1
        # -5：0
        for i in range(-self.FEATURES_REGION, 0):
            if sq_dist[i + self.FEATURES_REGION] > 0.05:
                break
            cloud_neighbors_picked[ind + i] = 1

        return cloud_neighbors_picked
    # 非可靠点过滤
    '''
    1.基于扫描线进行遍历，采用滑动固定窗口，注意边界;不是从第一个点开始+wind/2，和最后一个点-wind/2
    2.对当前扫描线内，相邻点夹角为正（dot>0.1）判断距离; 防止点过密
    3.基于点到传感器距离进行 赋权距离计算 ，仅考虑 赋权距离小于0.1的当前点的前5后1个点 
    '''
    def remove_unreliable(self, cloud_neighbors_picked, pcd, scan_start, scan_end):
        for i in range(scan_end.shape[0]):  # 0-max ring number
            sp = scan_start[i] + self.FEATURES_REGION  # 区间范围，兼顾前后窗口
            ep = scan_end[i] - self.FEATURES_REGION

            if ep - sp < self.N_SEGMENTS:  # 当前扫描线数量足够拆分为 N_SEGMENTS 个子段，
                continue

            for j in range(sp + 1, ep):
                prev_point = pcd[j - 1][:3]  # 上一点
                point = pcd[j][:3]  # 当前点
                next_point = pcd[j + 1][:3]  # 下一点
                # 当前点到下一个点的夹角，通过夹角来代替距离
                diff_next = np.dot(point - next_point, point - next_point)  #
                # 两点构成向量的点乘，夹角>0;因为雷达扫描基本是有序的，大概率不会出现负值[90,180]的情况;进行下一步判断，防止特征点过于密集
                if diff_next > 0.1:
                    depth1 = np.linalg.norm(point)  # 点到到传感器距离
                    depth2 = np.linalg.norm(next_point)  # 到传感器距离
                    # 小的做权重分母，大的做归一化分母
                    if depth1 > depth2:  # 当point的距离远时，用一个小于1的权重乘next_point，增大距离差。
                        weighted_dist = np.sqrt(np.dot(point - next_point * depth2 / depth1,
                                                       point - next_point * depth2 / depth1)) / depth2
                        if weighted_dist < 0.1:
                            cloud_neighbors_picked[j - self.FEATURES_REGION:j + 1] = 1  # 前5后1,7个点
                            #sss=sum(cloud_neighbors_picked)
                            #self.selectedPC(pcd, cloud_neighbors_picked)
                            continue
                    else:
                        weighted_dist = np.sqrt(np.dot(point - next_point * depth1 / depth2,
                                                       point - next_point * depth1 / depth2)) / depth1

                        if weighted_dist < 0.1:
                            cloud_neighbors_picked[j - self.FEATURES_REGION: j + self.FEATURES_REGION + 1] = 1  # 前5 后6
                            #sss = sum(cloud_neighbors_picked)
                            #self.selectedPC(pcd, cloud_neighbors_picked)
                            continue
                    # 向前判断，(|P1||P2|cos Theta)/ |P1|>0.0002
                    diff_prev = np.dot(point - prev_point, point - prev_point)
                    dis = np.dot(point, point)
                    # 上一个点和下一个点到当前点的
                    if diff_next > 0.0002 * dis and diff_prev > 0.0002 * dis:
                        cloud_neighbors_picked[j] = 1
                        #sss = sum(cloud_neighbors_picked)
                        #self.selectedPC(pcd, cloud_neighbors_picked)
        #self.selectedPC(pcd, cloud_neighbors_picked)
        return cloud_neighbors_picked

    # （仅针对面特征）排除可能存在遮挡区域的特征，判断条件为 当前点与附近点（一个segment） 不存在巨大间距 0.54 m。
    def has_gap(self, laser_cloud, ind):
        diff_S = laser_cloud[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION + 1, :3] - laser_cloud[ind, :3]  # seg点-当前点
        sq_dist = matrix_dot_product(diff_S[:, :3], diff_S[:, :3])  # sq_dist=x^2+y^2+z^2
        gapped = sq_dist[sq_dist > 0.3]  # 距离到当前点ind大于 0.548m=sqrt(0.3) 的点/ 夹角阈值
        if gapped.shape[0] > 0:  # 存在间隙，潜在不稳定
            return True
        else:
            return False
    # （仅针对边特征）也是基于距离来判断前后点
    def can_be_edge(self, laser_cloud, ind):
        #curpc=laser_cloud[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION, :3]
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(curpc[:, 0:3])
        #pc_show([pcd])
        diff_S = laser_cloud[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION, :3] -\
                 laser_cloud[ind - self.FEATURES_REGION + 1:ind + self.FEATURES_REGION + 1, :3]  # 当前点-下一个点，错位相减 p(k)-p(k+1)
        sq_dist = matrix_dot_product(diff_S[:, :3], diff_S[:, :3])  #
        gapped = laser_cloud[ind - self.FEATURES_REGION:ind + self.FEATURES_REGION, :3][sq_dist > 0.2]  # x^2+y^2+z^2>0.2的点
        if len(gapped) == 0:  # 夹角阈值都不超过 0.2
            return True
        else:
            # a=np.linalg.norm(gapped, axis=1)  # 大于阈值点的范数
            # b=np.linalg.norm(laser_cloud[ind][:3])  # 当前点的1-范数
            return np.any(np.linalg.norm(gapped, axis=1) > np.linalg.norm(laser_cloud[ind][:3]))
