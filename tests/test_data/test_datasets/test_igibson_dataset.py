# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet3d.datasets import IGibsonDataset


def _generate_sunrgbd_dataset_config():
    root_path = './tests/data/igibson'
    # in coordinate system refactor, this test file is modified
    ann_file = './tests/data/igibson/igibson_infos.pkl'
    class_names = ('basket', 'bathtub', 'bed', 'bench', 'bottom_cabinet',
                        'bottom_cabinet_no_top', 'carpet', 'chair', 'chest',
                        'coffee_machine', 'coffee_table', 'console_table',
                        'cooktop', 'counter', 'crib', 'cushion', 'dishwasher',
                        'door', 'dryer', 'fence', 'floor_lamp', 'fridge',
                        'grandfather_clock', 'guitar', 'heater', 'laptop',
                        'loudspeaker', 'microwave', 'mirror', 'monitor',
                        'office_chair', 'oven', 'piano', 'picture', 'plant',
                        'pool_table', 'range_hood', 'shelf', 'shower', 'sink',
                        'sofa', 'sofa_chair', 'speaker_system', 'standing_tv',
                        'stool', 'stove', 'table', 'table_lamp', 'toilet',
                        'top_cabinet', 'towel_rack', 'trash_can', 'treadmill',
                        'wall_clock', 'wall_mounted_tv', 'washer', 'window')
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=False,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(type='LoadAnnotations3D'),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=0.5,
        ),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.523599, 0.523599],
            scale_ratio_range=[0.85, 1.15],
            shift_height=False),
        dict(type='PointSample', num_points=5),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
            meta_keys=[
                'file_name', 'pcd_horizontal_flip', 'sample_idx',
                'pcd_scale_factor', 'pcd_rotation'
            ]),
    ]
    modality = dict(use_lidar=True, use_camera=False)
    return root_path, ann_file, class_names, pipelines, modality


def _generate_sunrgbd_multi_modality_dataset_config():
    root_path = './tests/data/sunrgbd'
    ann_file = './tests/data/sunrgbd/sunrgbd_infos.pkl'
    class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                   'dresser', 'night_stand', 'bookshelf', 'bathtub')
    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
    pipelines = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations3D'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(1333, 600), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(
            type='RandomFlip3D',
            sync_2d=False,
            flip_ratio_bev_horizontal=0.5,
        ),
        dict(
            type='GlobalRotScaleTrans',
            rot_range=[-0.523599, 0.523599],
            scale_ratio_range=[0.85, 1.15],
            shift_height=True),
        dict(type='PointSample', num_points=5),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(
            type='Collect3D',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'points', 'gt_bboxes_3d',
                'gt_labels_3d'
            ])
    ]
    modality = dict(use_lidar=True, use_camera=True)
    return root_path, ann_file, class_names, pipelines, modality


def test_getitem():

    from os import path as osp

    np.random.seed(0)
    root_path, ann_file, class_names, pipelines, modality = \
        _generate_sunrgbd_dataset_config()

    ig_dataset = IGibsonDataset(
        root_path, ann_file, pipelines, modality=modality)
    data = ig_dataset[0]
    print(f'data.keys: {data.keys()}')
    points = data['points']._data
    print(f'points.shape: {points.shape}')
    gt_bboxes_3d = data['gt_bboxes_3d']._data
    print(f'gt_bboxes_3d.shape: {gt_bboxes_3d.tensor.shape}')
    gt_labels_3d = data['gt_labels_3d']._data
    print(f'gt_labels_3d.shape: {gt_labels_3d.shape}')
    file_name = data['img_metas']._data['file_name']
    pcd_horizontal_flip = data['img_metas']._data['pcd_horizontal_flip']
    pcd_scale_factor = data['img_metas']._data['pcd_scale_factor']
    pcd_rotation = data['img_metas']._data['pcd_rotation']
    sample_idx = data['img_metas']._data['sample_idx']
    pcd_rotation_expected = np.array([[0.99889565, 0.04698427, 0.],
                                      [-0.04698427, 0.99889565, 0.],
                                      [0., 0., 1.]])
    expected_file_name = osp.join('./tests/data/igibson', 'points/Beechwood_0_int_00000.bin')
    assert file_name == expected_file_name
    assert pcd_horizontal_flip is False
    assert abs(pcd_scale_factor - 0.9770964398016714) < 1e-5
    assert np.allclose(pcd_rotation, pcd_rotation_expected, 1e-3)
    assert sample_idx == 'Beechwood_0_int_00000'
    expected_points = torch.tensor([[-0.9904, 1.2596, 0.1105, 0.0905],
                                    [-0.9948, 1.2758, 0.0437, 0.0238],
                                    [-0.9866, 1.2641, 0.0504, 0.0304],
                                    [-0.9915, 1.2586, 0.1265, 0.1065],
                                    [-0.9890, 1.2561, 0.1216, 0.1017]])
    expected_gt_bboxes_3d = torch.tensor(
        [[0.8308, 4.1168, -1.2035, 2.2493, 1.8444, 1.9245, 1.6486],
         [2.3002, 4.8149, -1.2442, 0.5718, 0.8629, 0.9510, 1.6030],
         [-1.1477, 1.8090, -1.1725, 0.6965, 1.5273, 2.0563, 0.0552]])
    expected_gt_bboxes_3d = torch.tensor(
        [[ 7.30000973e-01, -3.35495281e+00, -1.00060749e+00,
           1.39999998e+00,  4.69999999e-01,  1.18799996e+00,
          -3.14159231e+00],
         [ 7.45684147e-01, -3.05516863e+00, -1.29990971e+00,
           7.30000019e-01,  2.89999992e-01,  5.93999982e-01,
          -0.00000000e+00],
         [-2.37606859e+00,  3.49611282e-01, -1.17529869e+00,
           7.05186009e-01,  6.39160991e-01,  8.41499984e-01,
          -7.81321999e-01],
         [ 6.22774601e-01,  4.20982838e-01, -1.17598832e+00,
           6.24122024e-01,  6.65561974e-01,  8.41499984e-01,
           5.99119869e-01],
         [ 3.04924965e-01, -7.01088905e-02, -1.39971364e+00,
           5.64500988e-01,  4.85632002e-01,  3.95999998e-01,
           6.02354002e-01],
         [-6.44999981e-01,  1.11000061e+00, -1.22928405e+00,
           1.61000001e+00,  7.20000029e-01,  7.42500007e-01,
           1.21723104e-07],
         [ 1.38060427e+00,  5.69450855e-01, -7.25305915e-01,
           5.19999981e-01,  4.60000008e-01,  1.74240005e+00,
           1.06937281e-04],
         [ 1.20004082e+00,  1.00204468e-01, -1.34966242e+00,
           4.00000006e-01,  3.60000014e-01,  4.95000005e-01,
           1.97391378e-01],
         [ 2.77500010e+00, -9.66769695e-01, -1.59649968e+00,
           1.80999994e+00,  1.25000000e+00,  6.93000015e-03,
          -0.00000000e+00],
         [-1.43258047e+00, -1.37495422e+00, -1.59000003e+00,
           2.28999996e+00,  1.71000004e+00,  1.97999999e-02,
          -1.58956157e+00],
         [-3.65000248e-01, -3.58999944e+00,  1.25000119e-01,
           6.70000017e-01,  1.99999996e-02,  4.45499986e-01,
          -3.14159231e+00],
         [-5.60233593e-01,  1.55473804e+00, -4.00251746e-01,
           1.84000003e+00,  1.00000001e-01,  1.98000002e+00,
          -0.00000000e+00],
         [-1.63499975e+00, -3.64631557e+00, -4.98449802e-01,
           1.66999996e+00,  9.00000036e-02,  2.17799997e+00,
          -0.00000000e+00],
         [-2.93512678e+00,  2.45376587e-01, -3.99999976e-01,
           8.39999974e-01,  9.00000036e-02,  1.98000002e+00,
          -1.57210219e+00],
         [-2.93474722e+00, -2.93467879e+00, -3.99996281e-01,
           8.39999974e-01,  9.00000036e-02,  1.98000002e+00,
          -1.57210219e+00]])
    # coord sys refactor (rotation is correct but yaw has to be reversed)
    expected_gt_bboxes_3d[:, 6:] = -expected_gt_bboxes_3d[:, 6:]
    expected_gt_labels = np.array([32,  7, 41,  7, 44, 46, 20, 46,  6,  6, 33, 56, 17, 56, 56])
    original_classes = ig_dataset.CLASSES

    # assert torch.allclose(points, expected_points, 1e-2)
    # assert torch.allclose(gt_bboxes_3d.tensor, expected_gt_bboxes_3d, 1e-3)
    assert np.all(gt_labels_3d.numpy() == expected_gt_labels)
    assert original_classes == class_names

    ig_dataset = IGibsonDataset(
        root_path, ann_file, pipeline=None, classes=['bed', 'table'])
    assert ig_dataset.CLASSES != original_classes
    assert ig_dataset.CLASSES == ['bed', 'table']

    ig_dataset = IGibsonDataset(
        root_path, ann_file, pipeline=None, classes=('bed', 'table'))
    assert ig_dataset.CLASSES != original_classes
    assert ig_dataset.CLASSES == ('bed', 'table')

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + 'classes.txt'
        with open(path, 'w') as f:
            f.write('bed\ntable\n')

    ig_dataset = IGibsonDataset(
        root_path, ann_file, pipeline=None, classes=path)
    assert ig_dataset.CLASSES != original_classes
    assert ig_dataset.CLASSES == ['bed', 'table']



def test_evaluate():
    if not torch.cuda.is_available():
        pytest.skip()
    from mmdet3d.core.bbox.structures import DepthInstance3DBoxes
    root_path, ann_file, _, pipelines, modality = \
        _generate_sunrgbd_dataset_config()
    ig_dataset = IGibsonDataset(
        root_path, ann_file, pipelines, modality=modality)
    results = []
    pred_boxes = dict()
    pred_boxes['boxes_3d'] = DepthInstance3DBoxes(
        torch.tensor(
            [[ 7.30000973e-01, -3.35495281e+00, -1.00060749e+00, 1.39999998e+00,  4.69999999e-01,  1.18799996e+00, -3.14159231e+00],
             [2.5831, 4.8117, -1.2733, 0.5852, 0.8832, 0.9733, 1.6500],
             [-2.37606859e+00,  3.49611282e-01, -1.17529869e+00, 7.05186009e-01,  6.39160991e-01,  8.41499984e-01, -7.81321999e-01]]))
    pred_boxes['labels_3d'] = torch.tensor([32, 7, 41])
    pred_boxes['scores_3d'] = torch.tensor([0.9, 1.0, 1.0])
    results.append(pred_boxes)
    metric = [0.25, 0.5]
    ap_dict = ig_dataset.evaluate(results, metric)
    bed_precision_25 = ap_dict['piano_AP_0.25']
    dresser_precision_25 = ap_dict['chair_AP_0.25']
    night_stand_precision_25 = ap_dict['sofa_chair_AP_0.25']
    assert abs(bed_precision_25 - 1) < 0.01
    assert abs(dresser_precision_25 - 1) < 0.01
    assert abs(night_stand_precision_25 - 1) < 0.01


def test_show():
    import tempfile
    from os import path as osp

    import mmcv

    from mmdet3d.core.bbox import DepthInstance3DBoxes
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    root_path, ann_file, class_names, pipelines, modality = \
        _generate_sunrgbd_dataset_config()
    ig_dataset = IGibsonDataset(
        root_path, ann_file, pipelines, modality=modality)
    boxes_3d = DepthInstance3DBoxes(
        torch.tensor(
            [[1.1500, 4.2614, -1.0669, 1.3219, 2.1593, 1.0267, 1.6473],
             [-0.9583, 2.1916, -1.0881, 0.6213, 1.3022, 1.6275, -3.0720],
             [2.5697, 4.8152, -1.1157, 0.5421, 0.7019, 0.7896, 1.6712],
             [0.7283, 2.5448, -1.0356, 0.7691, 0.9056, 0.5771, 1.7121],
             [-0.9860, 3.2413, -1.2349, 0.5110, 0.9940, 1.1245, 0.3295]]))
    scores_3d = torch.tensor(
        [1.5280e-01, 1.6682e-03, 6.2811e-04, 1.2860e-03, 9.4229e-06])
    labels_3d = torch.tensor([0, 0, 0, 0, 0])
    result = dict(boxes_3d=boxes_3d, scores_3d=scores_3d, labels_3d=labels_3d)
    results = [result]
    ig_dataset.show(results, temp_dir, show=False)
    pts_file_path = osp.join(temp_dir, 'Beechwood_0_int_00000', 'Beechwood_0_int_00000_points.obj')
    gt_file_path = osp.join(temp_dir, 'Beechwood_0_int_00000', 'Beechwood_0_int_00000_gt.obj')
    pred_file_path = osp.join(temp_dir, 'Beechwood_0_int_00000', 'Beechwood_0_int_00000_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()

    # test show with pipeline
    eval_pipeline = [
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=True,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(
            type='DefaultFormatBundle3D',
            class_names=class_names,
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ]
    tmp_dir = tempfile.TemporaryDirectory()
    temp_dir = tmp_dir.name
    ig_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    pts_file_path = osp.join(temp_dir, 'Beechwood_0_int_00000', 'Beechwood_0_int_00000_points.obj')
    gt_file_path = osp.join(temp_dir, 'Beechwood_0_int_00000', 'Beechwood_0_int_00000_gt.obj')
    pred_file_path = osp.join(temp_dir, 'Beechwood_0_int_00000', 'Beechwood_0_int_00000_pred.obj')
    mmcv.check_file_exist(pts_file_path)
    mmcv.check_file_exist(gt_file_path)
    mmcv.check_file_exist(pred_file_path)
    tmp_dir.cleanup()

    # # test multi-modality show
    # tmp_dir = tempfile.TemporaryDirectory()
    # temp_dir = tmp_dir.name
    # root_path, ann_file, class_names, multi_modality_pipelines, modality = \
    #     _generate_sunrgbd_multi_modality_dataset_config()
    # ig_dataset = IGibsonDataset(
    #     root_path, ann_file, multi_modality_pipelines, modality=modality)
    # ig_dataset.show(results, temp_dir, False, multi_modality_pipelines)
    # pts_file_path = osp.join(temp_dir, '000001', '000001_points.obj')
    # gt_file_path = osp.join(temp_dir, '000001', '000001_gt.obj')
    # pred_file_path = osp.join(temp_dir, '000001', '000001_pred.obj')
    # img_file_path = osp.join(temp_dir, '000001', '000001_img.png')
    # img_pred_path = osp.join(temp_dir, '000001', '000001_pred.png')
    # img_gt_file = osp.join(temp_dir, '000001', '000001_gt.png')
    # mmcv.check_file_exist(pts_file_path)
    # mmcv.check_file_exist(gt_file_path)
    # mmcv.check_file_exist(pred_file_path)
    # mmcv.check_file_exist(img_file_path)
    # mmcv.check_file_exist(img_pred_path)
    # mmcv.check_file_exist(img_gt_file)
    # tmp_dir.cleanup()

    # # test multi-modality show with pipeline
    # eval_pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     dict(
    #         type='LoadPointsFromFile',
    #         coord_type='DEPTH',
    #         shift_height=True,
    #         load_dim=6,
    #         use_dim=[0, 1, 2]),
    #     dict(
    #         type='DefaultFormatBundle3D',
    #         class_names=class_names,
    #         with_label=False),
    #     dict(type='Collect3D', keys=['points', 'img'])
    # ]
    # tmp_dir = tempfile.TemporaryDirectory()
    # temp_dir = tmp_dir.name
    # ig_dataset.show(results, temp_dir, show=False, pipeline=eval_pipeline)
    # pts_file_path = osp.join(temp_dir, '000001', '000001_points.obj')
    # gt_file_path = osp.join(temp_dir, '000001', '000001_gt.obj')
    # pred_file_path = osp.join(temp_dir, '000001', '000001_pred.obj')
    # img_file_path = osp.join(temp_dir, '000001', '000001_img.png')
    # img_pred_path = osp.join(temp_dir, '000001', '000001_pred.png')
    # img_gt_file = osp.join(temp_dir, '000001', '000001_gt.png')
    # mmcv.check_file_exist(pts_file_path)
    # mmcv.check_file_exist(gt_file_path)
    # mmcv.check_file_exist(pred_file_path)
    # mmcv.check_file_exist(img_file_path)
    # mmcv.check_file_exist(img_pred_path)
    # mmcv.check_file_exist(img_gt_file)
    # tmp_dir.cleanup()

if __name__ == '__main__':
    test_getitem()
    # test_evaluate()
    # test_show()