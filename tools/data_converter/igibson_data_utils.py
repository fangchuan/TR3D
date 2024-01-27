"""
Copyright (c) 2024 Chuan FANG, Hong Kong Unicersity of Science and Technology.
email: cfangac@connect.ust.hk
"""
from concurrent import futures as futures
from os import path as osp

import mmcv
import numpy as np
from scipy import io as sio


def random_sampling(points, num_points, replace=None, return_choices=False):
    """Random sampling.

    Sampling point cloud to a certain number of points.

    Args:
        points (ndarray): Point cloud.
        num_points (int): The number of samples.
        replace (bool): Whether the sample is with or without replacement.
        return_choices (bool): Whether to return choices.

    Returns:
        points (ndarray): Point cloud after sampling.
    """

    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]

def read_split_list(list_file):
    import json
    pkl_list = []
    with open(list_file) as f:
        lines = json.load(f)
        for line in lines:
            pkl_name = line.split("/")
            pkl_list.append(pkl_name[0]+"_"+pkl_name[1])
    return pkl_list


class IGibsonObjectInstance(object):

    def __init__(self, bbox:np.array, label2cat:dict):
        semantic_class = bbox[7]
        self.classname = label2cat[semantic_class]
        # self.xmin = data[1]
        # self.ymin = data[2]
        # self.xmax = data[1] + data[3]
        # self.ymax = data[2] + data[4]
        # self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.centroid = bbox[0:3]
        self.width = bbox[4]
        self.length = bbox[3]
        self.height = bbox[5]
        # data[9] is x_size (length), data[8] is y_size (width), data[10] is
        # z_size (height) in our depth coordinate system,
        # l corresponds to the size along the x axis
        self.size = (bbox[3:6]) * 2
        self.orientation = np.zeros((3, ))
        # self.orientation[0] = data[11]
        # self.orientation[1] = data[12]
        # self.heading_angle = np.arctan2(self.orientation[1],
        #                                 self.orientation[0])
        self.heading_angle = bbox[6]

        corners = self.get_corners(self.centroid, self.size, self.heading_angle)
        # self.centroid = np.mean(corners, axis=0)
        self.box3d = np.concatenate(
            [self.centroid, self.size, self.heading_angle[None]])


    def rotz(self,t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
        
    def get_corners(self, center, size, heading_angle):
        R = self.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

class IGibsonData(object):
    """Igibson-synthetic data.

    Generate scannet infos for igibson_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        use_v1 (bool, optional): Whether to use v1. Default: False.
    """

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path, 'panocontext_traintest')
        self.cat2label={'basket':0, 'bathtub':1, 'bed':2, 'bench':3, 'bottom_cabinet':4,
                        'bottom_cabinet_no_top':5, 'carpet':6, 'chair':7, 'chest':8,
                        'coffee_machine':9, 'coffee_table':10, 'console_table':11,
                        'cooktop':12, 'counter':13, 'crib':14, 'cushion':15, 'dishwasher':16,
                        'door':17, 'dryer':18, 'fence':19, 'floor_lamp':20, 'fridge':21,
                        'grandfather_clock':22, 'guitar':23, 'heater':24, 'laptop':25,
                        'loudspeaker':26, 'microwave':27, 'mirror':28, 'monitor':29,
                        'office_chair':30, 'oven':31, 'piano':32, 'picture':33, 'plant':34,
                        'pool_table':35, 'range_hood':36, 'shelf':37, 'shower':38, 'sink':39,
                        'sofa':40, 'sofa_chair':41, 'speaker_system':42, 'standing_tv':43,
                        'stool':44, 'stove':45, 'table':46, 'table_lamp':47, 'toilet':48,
                        'top_cabinet':49, 'towel_rack':50, 'trash_can':51, 'treadmill':52,
                        'wall_clock':53, 'wall_mounted_tv':54, 'washer':55, 'window':56}
        self.label2cat = {v:k for k, v in self.cat2label.items()}
        
        assert split in ['train', 'test']
        split_file = osp.join(self.split_dir, f'{split}.json')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = read_split_list(split_file)
        # print(f'{split} sample_id_list: {self.sample_id_list}')
        self.image_dir = osp.join(self.split_dir, 'igibson_image')
        # self.calib_dir = osp.join(self.split_dir, 'calib')
        self.depth_dir = osp.join(self.split_dir, 'igibson_depth')
        self.label_dir = osp.join(self.split_dir, 'igibson_3dbbox')
        
        self.fibonacci_mask = self.get_fibonacci_mask(nb_samples=60000)
        self.unit_map = self.get_unit_map()

    def __len__(self):
        return len(self.sample_id_list)

    def get_image(self, idx):
        img_filename = osp.join(self.image_dir, f'{idx}.png')
        return mmcv.imread(img_filename)

    def get_image_shape(self, idx):
        image = self.get_image(idx)
        return np.array(image.shape[:2], dtype=np.int32)

    def get_depth(self, idx):
        """
        idx: scan namme
        """
        depth_filename = osp.join(self.depth_dir, f'{idx}_depth_pred.npy')
        depth = np.load(depth_filename)
        return depth

    def get_calibration(self, idx):
        calib_filepath = osp.join(self.calib_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rt = np.array([float(x) for x in lines[0].split(' ')])
        Rt = np.reshape(Rt, (3, 3), order='F').astype(np.float32)
        K = np.array([float(x) for x in lines[1].split(' ')])
        K = np.reshape(K, (3, 3), order='F').astype(np.float32)
        return K, Rt

    def get_label_objects(self, idx):
        label_filename = osp.join(self.label_dir, f'{idx}_bbox.npy')
        # lines = [line.rstrip() for line in open(label_filename)]
        obj_3dbboxes = np.load(label_filename)
        objects = [IGibsonObjectInstance(bbox, self.label2cat) for bbox in obj_3dbboxes]
        
        return objects

    def get_unit_map(self):
        h = 512
        w = 1024
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = np.repeat(Phi, h, axis=0)

        X = np.expand_dims(np.sin(Theta) * np.sin(Phi),2)
        Y =  np.expand_dims(np.cos(Theta),2)
        Z = np.expand_dims(np.sin(Theta) * np.cos(Phi),2)
        unit_map = np.concatenate([X,Z,Y],axis=2)

        return unit_map

    def get_fibonacci_mask(self,nb_samples):
        sampled_point = self.fibonacci_spiral_samples_on_unit_sphere(nb_samples=nb_samples)
        mask = np.zeros((512,1024))
        for sampled_point_i in range(sampled_point.shape[0]):
            xyz = sampled_point[sampled_point_i, :]
            uv = self.unitxyz2uv(xyz, equ_w=1024, equ_h=512)
            mask[int(uv[1]), int(uv[0])] = 1
        mask = np.where(mask > 0.5, True, False)
        return mask

    def unitxyz2uv(self, xyz, equ_w, equ_h, normlized = False):
        x, z, y = np.split(xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        if normlized:
            u = (lon / (2 * np.pi) + 0.5)
            v = (-lat / np.pi + 0.5)
        else:
            u = (lon / (2 * np.pi) + 0.5) * equ_w - 0.5
            v = (-lat / np.pi + 0.5) * equ_h - 0.5
        return [u, v]

    def fibonacci_spiral_samples_on_unit_sphere(self, nb_samples, mode=0):
        shift = 1.0 if mode == 0 else nb_samples * np.random.random()

        ga = np.pi * (3.0 - np.sqrt(5.0))
        offset = 2.0 / nb_samples

        ss = np.zeros((nb_samples, 3))
        j = 0
        for i in range(nb_samples):
            phi = ga * ((i + shift) % nb_samples)
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = ((i + 0.5) * offset) - 1.0
            sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
            ss[j, :] = np.array([sin_phi * sin_theta, cos_theta, cos_phi * sin_theta])
            j += 1
        return ss
    
    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None, show_results=True):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            
            current_mask = self.fibonacci_mask.copy()
            
            info = dict()
            
            img_path = osp.join(self.image_dir, f'{sample_idx}.png')
            rgb = self.get_image(sample_idx)[:, :, (2, 1, 0)]
            image_info = {
                'image_idx': sample_idx,
                'image_shape': rgb.shape[:2],
                'image_path': img_path
            }
            info['image'] = image_info
            
            # convert depth to points
            SAMPLE_NUM = 50000
            # TODO: Check whether can move the point
            #  sampling process during training.
            depth = self.get_depth(sample_idx)
            point_cloud_map = np.repeat(np.expand_dims(depth,axis=2),3,axis=2)*self.unit_map
            point_cloud = point_cloud_map[current_mask]
            point_cloud_rgb = rgb[current_mask]
            # point_cloud_rgb = (point_cloud_rgb-MEAN_COLOR_RGB)/MEAN_COLOR_STD
            point_cloud = np.concatenate([point_cloud,point_cloud_rgb],axis=1)
            
            print(f'point_cloud.shape: {point_cloud.shape}')
            pc_upright_depth_subsampled = random_sampling(
                point_cloud, SAMPLE_NUM)
            print(f'pc_upright_depth_subsampled.shape: {pc_upright_depth_subsampled.shape}')
            
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # save points
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            pc_upright_depth_subsampled.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))

            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            # K, Rt = self.get_calibration(sample_idx)
            # calib_info = {'K': K, 'Rt': Rt}
            info['calib'] = None

            if has_label:
                obj_list = self.get_label_objects(sample_idx)
                annotations = {}
                annotations['gt_num'] = len([
                    obj.classname for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ])
                if annotations['gt_num'] != 0:
                    annotations['name'] = np.array([
                        obj.classname for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    # annotations['bbox'] = np.concatenate([
                    #     obj.box2d.reshape(1, 4) for obj in obj_list
                    #     if obj.classname in self.cat2label.keys()
                    # ], axis=0)
                    annotations['bbox'] = None
                    annotations['location'] = np.concatenate([
                        obj.centroid.reshape(1, 3) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ], axis=0)
                    annotations['dimensions'] = 2 * np.array([
                        [obj.length, obj.width, obj.height] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])  # lwh (depth) format
                    annotations['rotation_y'] = np.array([
                        obj.heading_angle for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['index'] = np.arange(len(obj_list), dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat2label[obj.classname] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['gt_boxes_upright_depth'] = np.stack(
                        [
                            obj.box3d for obj in obj_list
                            if obj.classname in self.cat2label.keys()
                        ],
                        axis=0)  # (K,7)
                if show_results:
                    from mmdet3d.core import ( show_result)
                    gt_bboxes_3d = annotations['gt_boxes_upright_depth']
                    print(f'gt_bboxes_3d.shape: {gt_bboxes_3d.shape}')
                    show_result(
                        points=pc_upright_depth_subsampled,
                        gt_bboxes=gt_bboxes_3d,
                        pred_bboxes=None,
                        show=False,
                        out_dir=osp.join(self.root_dir, 'data_vis'),
                        filename=sample_idx,
                        snapshot=False,
                        pred_labels=None)
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if \
            sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
