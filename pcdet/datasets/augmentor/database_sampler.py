import pickle

import os
import copy
import numpy as np
from skimage import io
import torch
import SharedArray
import torch.distributed as dist
import cv2
from copy import deepcopy
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common
from PIL import Image

from .cross_modal_augmentation import *

class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg

        #NuScenes
        self.frustum_double_mask = sampler_cfg.get('FRUSTUM_DOUBLE_MASK', False)
        self.img_aug_type = sampler_cfg.get('IMG_AUG_TYPE', None)
        self.img_aug_iou_thresh = sampler_cfg.get('IMG_AUG_IOU_THRESH', 0.5)
        # Mixup
        self.img_aug_mixup = sampler_cfg.get('IMG_AUG_MIXUP', 0.7)

        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)

        #TODO: 提前生成了用于 GT Sampling 的数据库
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            if not db_info_path.exists():
                assert len(sampler_cfg.DB_INFO_PATH) == 1
                sampler_cfg.DB_INFO_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_INFO_PATH']
                sampler_cfg.DB_DATA_PATH[0] = sampler_cfg.BACKUP_DB_INFO['DB_DATA_PATH']
                db_info_path = self.root_path.resolve() / sampler_cfg.DB_INFO_PATH[0]
                sampler_cfg.NUM_POINT_FEATURES = sampler_cfg.BACKUP_DB_INFO['NUM_POINT_FEATURES']

            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None
        if self.img_aug_type == 'nuscenes':
            self.gt_database_data_key_img = self.load_db_to_shared_memory_img() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        # assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]        
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key
    
    def load_db_to_shared_memory_img(self):
        self.logger.info('Loading img GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        # assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path_img = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[1]
        

        sa_key = self.sampler_cfg.DB_DATA_PATH[1]
        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path_img)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('img GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict


    #TODO: used for KITTI
    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def copy_paste_to_image_kitti(self, data_dict, crop_feat, gt_number, point_idxes=None):
        kitti_img_aug_type = 'by_depth'
        kitti_img_aug_use_type = 'annotation'

        image = data_dict['images']
        boxes3d = data_dict['gt_boxes']
        boxes2d = data_dict['gt_boxes2d']
        corners_lidar = box_utils.boxes_to_corners_3d(boxes3d)
        if 'depth' in kitti_img_aug_type:
            paste_order = boxes3d[:,0].argsort()
            paste_order = paste_order[::-1]
        else:
            paste_order = np.arange(len(boxes3d),dtype=np.int)

        if 'reverse' in kitti_img_aug_type:
            paste_order = paste_order[::-1]

        paste_mask = -255 * np.ones(image.shape[:2], dtype=np.int)
        fg_mask = np.zeros(image.shape[:2], dtype=np.int)
        overlap_mask = np.zeros(image.shape[:2], dtype=np.int)
        depth_mask = np.zeros((*image.shape[:2], 2), dtype=np.float)
        points_2d, depth_2d = data_dict['calib'].lidar_to_img(data_dict['points'][:,:3])
        points_2d[:,0] = np.clip(points_2d[:,0], a_min=0, a_max=image.shape[1]-1)
        points_2d[:,1] = np.clip(points_2d[:,1], a_min=0, a_max=image.shape[0]-1)
        points_2d = points_2d.astype(np.int)
        for _order in paste_order:
            _box2d = boxes2d[_order]
            image[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = crop_feat[_order]
            overlap_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] += \
                (paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] > 0).astype(np.int)
            paste_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = _order

            if 'cover' in kitti_img_aug_use_type:
                # HxWx2 for min and max depth of each box region
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],0] = corners_lidar[_order,:,0].min()
                depth_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2],1] = corners_lidar[_order,:,0].max()

            # foreground area of original point cloud in image plane
            if _order < gt_number:
                fg_mask[_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = 1

        data_dict['images'] = image

        # if not self.joint_sample:
        #     return data_dict

        new_mask = paste_mask[points_2d[:,1], points_2d[:,0]]==(point_idxes+gt_number)
        if False:  # self.keep_raw:
            raw_mask = (point_idxes == -1)
        else:
            raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < gt_number)
            raw_bg = (fg_mask == 0) & (paste_mask < 0)
            raw_mask = raw_fg[points_2d[:,1], points_2d[:,0]] | raw_bg[points_2d[:,1], points_2d[:,0]]
        keep_mask = new_mask | raw_mask
        data_dict['points_2d'] = points_2d

        if 'annotation' in kitti_img_aug_use_type:
            data_dict['points'] = data_dict['points'][keep_mask]
            data_dict['points_2d'] = data_dict['points_2d'][keep_mask]
        elif 'projection' in kitti_img_aug_use_type:
            overlap_mask[overlap_mask>=1] = 1
            data_dict['overlap_mask'] = overlap_mask
            if 'cover' in kitti_img_aug_use_type:
                data_dict['depth_mask'] = depth_mask

        return data_dict
    
    #TODO: callee : data_dict = self.copy_paste_to_image_nuscenes(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'])
    def copy_paste_to_image_nuscenes(self, data_dict, crop_feat, gt_number, point_idxes=None):
        nuscenes_img_aug_type = 'by_depth'

        image = data_dict['ori_imgs']
        boxes3d = data_dict['gt_boxes']
        boxes2d = data_dict['gt_boxes2d']
        # raw_img = deepcopy(image)

        # 已经写死了，先把远处的拼上去，再把近的拼上去
        if 'depth' in nuscenes_img_aug_type:
            paste_order = boxes3d[:,0].argsort()
            paste_order = paste_order[::-1]
        else:
            paste_order = np.arange(len(boxes3d),dtype=np.int)

        if 'reverse' in nuscenes_img_aug_type:
            paste_order = paste_order[::-1]
        boxes2d = boxes2d.astype(np.int32)

        #FIXME: 需要明确这里的mixup的具体操作，其实是把crop_image和原始图像进行融合，但是只有crop_image的一部分会被融合到原始图像中
        for _order in paste_order:
            _box2d = boxes2d[_order]
            idx = _box2d[-1]
            h = _box2d[3] - _box2d[1]
            w = _box2d[2] - _box2d[0]

            #TODO: in init: self.img_aug_mixup = sampler_cfg.get('IMG_AUG_MIXUP', 0.7)
            #改动的关键
            image[idx][_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]] = self.img_aug_mixup* crop_feat[_order][:h,:w] + \
                (1-self.img_aug_mixup)*image[idx][_box2d[1]:_box2d[3],_box2d[0]:_box2d[2]]
        
        
        # recover crop resize
        crop_imgs = []
        W, H = data_dict["ori_shape"]
        img_process_infos = data_dict['img_process_infos'] #这个是nuscenes自带的，用于处理图像的一些基本参数，便于放缩之类的
        for img, img_process_info in zip(image, img_process_infos):
            resize,crop = img_process_info[:2]
            img = Image.fromarray(img)
            resize_dims = (int(W * resize), int(H * resize))
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_imgs.append(img)

        data_dict['camera_imgs'] = crop_imgs
        return data_dict

    def collect_image_crops_kitti(self, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx):
        calib_file = kitti_common.get_calib_path(int(info['image_idx']), self.root_path, relative_path=False)
        sampled_calib = calibration_kitti.Calibration(calib_file)
        points_2d, depth_2d = sampled_calib.lidar_to_img(obj_points[:,:3])

        if True:  # self.point_refine:
            # align calibration metrics for points
            points_ract = data_dict['calib'].img_to_rect(points_2d[:,0], points_2d[:,1], depth_2d)
            points_lidar = data_dict['calib'].rect_to_lidar(points_ract)
            obj_points[:, :3] = points_lidar
            # align calibration metrics for boxes
            box3d_raw = sampled_gt_boxes[idx].reshape(1,-1)
            box3d_coords = box_utils.boxes_to_corners_3d(box3d_raw)[0]
            box3d_box, box3d_depth = sampled_calib.lidar_to_img(box3d_coords)
            box3d_coord_rect = data_dict['calib'].img_to_rect(box3d_box[:,0], box3d_box[:,1], box3d_depth)
            box3d_rect = box_utils.corners_rect_to_camera(box3d_coord_rect).reshape(1,-1)
            box3d_lidar = box_utils.boxes3d_kitti_camera_to_lidar(box3d_rect, data_dict['calib'])
            box2d = box_utils.boxes3d_kitti_camera_to_imageboxes(box3d_rect, data_dict['calib'],
                                                                    data_dict['images'].shape[:2])
            sampled_gt_boxes[idx] = box3d_lidar[0]
            sampled_gt_boxes2d[idx] = box2d[0]

        obj_idx = idx * np.ones(len(obj_points), dtype=np.int)

        # copy crops from images
        img_path = self.root_path /  f'training/image_2/{info["image_idx"]}.png'
        raw_image = io.imread(img_path)
        raw_image = raw_image.astype(np.float32)
        raw_center = info['bbox'].reshape(2,2).mean(0)
        new_box = sampled_gt_boxes2d[idx].astype(np.int)
        new_shape = np.array([new_box[2]-new_box[0], new_box[3]-new_box[1]])
        raw_box = np.concatenate([raw_center-new_shape/2, raw_center+new_shape/2]).astype(np.int)
        raw_box[0::2] = np.clip(raw_box[0::2], a_min=0, a_max=raw_image.shape[1])
        raw_box[1::2] = np.clip(raw_box[1::2], a_min=0, a_max=raw_image.shape[0])
        if (raw_box[2]-raw_box[0])!=new_shape[0] or (raw_box[3]-raw_box[1])!=new_shape[1]:
            new_center = new_box.reshape(2,2).mean(0)
            new_shape = np.array([raw_box[2]-raw_box[0], raw_box[3]-raw_box[1]])
            new_box = np.concatenate([new_center-new_shape/2, new_center+new_shape/2]).astype(np.int)

        img_crop2d = raw_image[raw_box[1]:raw_box[3],raw_box[0]:raw_box[2]] / 255

        return new_box, img_crop2d, obj_points, obj_idx

    def sample_gt_boxes_2d_kitti(self, data_dict, sampled_boxes, valid_mask):
        mv_height = None
        # filter out box2d iou > thres
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_boxes, data_dict['road_plane'], data_dict['calib']
            )

        # sampled_boxes2d = np.stack([x['bbox'] for x in sampled_dict], axis=0).astype(np.float32)
        boxes3d_camera = box_utils.boxes3d_lidar_to_kitti_camera(sampled_boxes, data_dict['calib'])
        sampled_boxes2d = box_utils.boxes3d_kitti_camera_to_imageboxes(boxes3d_camera, data_dict['calib'],
                                                                        data_dict['images'].shape[:2])
        sampled_boxes2d = torch.Tensor(sampled_boxes2d)
        existed_boxes2d = torch.Tensor(data_dict['gt_boxes2d'])
        iou2d1 = box_utils.pairwise_iou(sampled_boxes2d, existed_boxes2d).cpu().numpy()
        iou2d2 = box_utils.pairwise_iou(sampled_boxes2d, sampled_boxes2d).cpu().numpy()
        iou2d2[range(sampled_boxes2d.shape[0]), range(sampled_boxes2d.shape[0])] = 0
        iou2d1 = iou2d1 if iou2d1.shape[1] > 0 else iou2d2

        ret_valid_mask = ((iou2d1.max(axis=1)<self.img_aug_iou_thresh) &
                         (iou2d2.max(axis=1)<self.img_aug_iou_thresh) &
                         (valid_mask))

        sampled_boxes2d = sampled_boxes2d[ret_valid_mask].cpu().numpy()
        if mv_height is not None:
            mv_height = mv_height[ret_valid_mask]
        return sampled_boxes2d, mv_height, ret_valid_mask
    
    def sample_gt_boxes_2d_nuscenes(self, data_dict, sampled_boxes, valid_mask, sampled_dict):
        sampled_boxes2d = torch.Tensor(np.stack([x['box2d_camera'] for x in sampled_dict], axis=0).astype(np.float32))
        existed_boxes2d = torch.Tensor(data_dict['gt_boxes2d'])
        ret_valid_mask = np.zeros_like(valid_mask)
        for k in range(6):
            view_mask = sampled_boxes2d[:,-1] == k
            if view_mask.sum() == 0:
                continue
            view_sampled_boxes2d = sampled_boxes2d[view_mask][:,:4]
            pre_view_mask = existed_boxes2d[:,-1] == k
            view_existed_boxes2d = existed_boxes2d[pre_view_mask][:,:4]


            # # 计算选取的样本与已存在的样本之间的IoU 
            iou2d1 = box_utils.pairwise_iou(view_sampled_boxes2d, view_existed_boxes2d).cpu().numpy()
            iou2d2 = box_utils.pairwise_iou(view_sampled_boxes2d, view_sampled_boxes2d).cpu().numpy()
            iou2d2[range(view_sampled_boxes2d.shape[0]), range(view_sampled_boxes2d.shape[0])] = 0
            iou2d1 = iou2d1 if iou2d1.shape[1] > 0 else iou2d2

            # 更新ret_valid_mask，根据一些条件更新有效掩码  
            view_mask = view_mask.cpu().numpy().astype(bool)
            ret_valid_mask[view_mask] = ((iou2d1.max(axis=1)<self.img_aug_iou_thresh) &
                            (iou2d2.max(axis=1)<self.img_aug_iou_thresh) &
                            (valid_mask[view_mask]))

        # 根据ret_valid_mask对sampled_boxes2d进行过滤，并将其转换为NumPy数组  
        sampled_boxes2d = sampled_boxes2d[ret_valid_mask].cpu().numpy()
        return sampled_boxes2d, None, ret_valid_mask

    def sample_gt_boxes_2d(self, data_dict, sampled_boxes, valid_mask, sampled_dict=None):
        mv_height = None

        if self.img_aug_type == 'kitti':
            sampled_boxes2d, mv_height, ret_valid_mask = self.sample_gt_boxes_2d_kitti(data_dict, sampled_boxes, valid_mask)
        elif self.img_aug_type == 'nuscenes':
            sampled_boxes2d, mv_height, ret_valid_mask = self.sample_gt_boxes_2d_nuscenes(data_dict, sampled_boxes, valid_mask, sampled_dict)
        else:
            raise NotImplementedError

        return sampled_boxes2d, mv_height, ret_valid_mask

    #TODO: convert sampled 3D boxes to image plane
    def initilize_image_aug_dict(self, data_dict, gt_boxes_mask):
        img_aug_gt_dict = None
        if self.img_aug_type is None:
            pass
        elif self.img_aug_type == 'kitti':
            obj_index_list, crop_boxes2d = [], []
            gt_number = gt_boxes_mask.sum().astype(np.int)
            gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask].astype(np.int)
            gt_crops2d = [data_dict['images'][_x[1]:_x[3],_x[0]:_x[2]] for _x in gt_boxes2d]

            img_aug_gt_dict = {
                'obj_index_list': obj_index_list,
                'gt_crops2d': gt_crops2d,
                'gt_boxes2d': gt_boxes2d,
                'gt_number': gt_number,
                'crop_boxes2d': crop_boxes2d
            }
        elif self.img_aug_type == 'nuscenes':
            obj_index_list, crop_boxes2d = [], []
            gt_number = gt_boxes_mask.sum().astype(np.int)
            #FIXME: 需要搞懂gt_boxes2d是在哪一步产出的
            gt_boxes2d = data_dict['gt_boxes2d'][gt_boxes_mask].astype(np.int)
            gt_crops2d = [data_dict['ori_imgs'][_x[-1]][_x[1]:_x[3],_x[0]:_x[2]] for _x in gt_boxes2d]

            img_aug_gt_dict = {
                'obj_index_list': obj_index_list,
                'gt_crops2d': gt_crops2d,
                'gt_boxes2d': gt_boxes2d,
                'gt_number': gt_number,
                'crop_boxes2d': crop_boxes2d
            }
        else:
            raise NotImplementedError

        return img_aug_gt_dict

    def collect_image_crops(self, img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx, gt_database_data_img=None):
        if self.img_aug_type == 'kitti':
            new_box, img_crop2d, obj_points, obj_idx = self.collect_image_crops_kitti(info, data_dict,
                                                    obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx)
            img_aug_gt_dict['crop_boxes2d'].append(new_box)
            img_aug_gt_dict['gt_crops2d'].append(img_crop2d)
            img_aug_gt_dict['obj_index_list'].append(obj_idx)
        elif self.img_aug_type == 'nuscenes':
            if self.use_shared_memory:
                img_shape = info['img_shape']
                img_start_offset, img_end_offset = info['global_data_offset_img']
                img_crop2d = deepcopy(gt_database_data_img[img_start_offset:img_end_offset])[:,:3].reshape(img_shape)
            else:
                img_path = info['img_path']
                full_path = str(self.root_path /img_path)
                if not os.path.exists(full_path):
                    full_path = os.path.join('../data/',str(img_path))
                    if not os.path.exists(full_path):
                        print("cannot find img gt path: ",full_path)
                img_crop2d = cv2.imread(full_path)
            img_aug_gt_dict['gt_crops2d'].append(img_crop2d)
        else:
            raise NotImplementedError

        return img_aug_gt_dict, obj_points

    #TODO: called copy_paste_to_image_nuscenes
    # callee: copy_paste_to_image(img_aug_gt_dict, data_dict, points)
    def copy_paste_to_image(self, img_aug_gt_dict, data_dict, points):
        if self.img_aug_type == 'kitti':
            obj_points_idx = np.concatenate(img_aug_gt_dict['obj_index_list'], axis=0)
            point_idxes = -1 * np.ones(len(points), dtype=np.int)
            point_idxes[:obj_points_idx.shape[0]] = obj_points_idx

            data_dict['gt_boxes2d'] = np.concatenate([img_aug_gt_dict['gt_boxes2d'], np.array(img_aug_gt_dict['crop_boxes2d'])], axis=0)
            data_dict = self.copy_paste_to_image_kitti(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'], point_idxes)
            if 'road_plane' in data_dict:
                data_dict.pop('road_plane')
        elif self.img_aug_type == 'nuscenes':
            data_dict['gt_boxes2d'] = np.concatenate([img_aug_gt_dict['gt_boxes2d'], np.array(img_aug_gt_dict['crop_boxes2d'])], axis=0)
            data_dict = self.copy_paste_to_image_nuscenes(data_dict, img_aug_gt_dict['gt_crops2d'], img_aug_gt_dict['gt_number'])
        else:
            raise NotImplementedError
        return data_dict


    #TODO: callee of copy_paste_to_image
    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict, mv_height=None, sampled_gt_boxes2d=None):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False) and mv_height is None:
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []

        #TODO: convert sampled 3D boxes to image plane
        img_aug_gt_dict = self.initilize_image_aug_dict(data_dict, gt_boxes_mask)

        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
            if self.img_aug_type == 'nuscenes': 
                gt_database_data_img = SharedArray.attach(f"shm://{self.gt_database_data_key_img}")
                gt_database_data_img.setflags(write=0)
        else:
            gt_database_data = None
            if self.img_aug_type == 'nuscenes': 
                gt_database_data_img = None

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']

                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                if obj_points.shape[0] != info['num_points_in_gt']:
                    obj_points = np.fromfile(str(file_path), dtype=np.float64).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)
            
            if self.sampler_cfg.get('APPLY_TANH_DIM_LIST', False):
                for dim_idx in self.sampler_cfg.APPLY_TANH_DIM_LIST:
                    obj_points[:, dim_idx] = np.tanh(obj_points[:, dim_idx])

            assert obj_points.shape[0] == info['num_points_in_gt']
            obj_points[:, :3] += info['box3d_lidar'][:3].astype(np.float32)

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            #FIXME: 需要搞清楚image_aug_gt_dict是指什么
            if self.img_aug_type is not None:
                img_aug_gt_dict, obj_points = self.collect_image_crops(
                    img_aug_gt_dict, info, data_dict, obj_points, sampled_gt_boxes, sampled_gt_boxes2d, idx, gt_database_data_img
                )

            obj_points_list.append(obj_points)
        
        if self.img_aug_type == 'nuscenes':
            img_aug_gt_dict['crop_boxes2d'] = sampled_gt_boxes2d

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        # sampled_gt_points = np.array([x['points'] for x in total_valid_sampled_dict])

        if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) or obj_points.shape[-1] != points.shape[-1]:
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False):
                min_time = min(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
                max_time = max(self.sampler_cfg.TIME_RANGE[0], self.sampler_cfg.TIME_RANGE[1])
            else:
                assert obj_points.shape[-1] == points.shape[-1] + 1
                # transform multi-frame GT points to single-frame GT points
                min_time = max_time = 0.0

            time_mask = np.logical_and(obj_points[:, -1] < max_time + 1e-6, obj_points[:, -1] > min_time - 1e-6)
            obj_points = obj_points[time_mask]

        # large_sampled_gt_boxes = box_utils.enlarge_box3d(
        #     sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        # )
            
        # points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        # points = np.concatenate([obj_points[:, :points.shape[-1]], points], axis=0)

        #FIXME: modify in here
        # procress_points_modify(points,obj_points,data_dict)
        points = procress_points_modify(points,obj_points[:, :points.shape[-1]],data_dict)

        
        #TODO: 把确定可以加入的目标merge到当前的所有data_dict中
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names

        #TODO: PointAugmenting
        # points, gt_boxes_mask = procress_points(points, sampled_points, gt_boxes_mask, gt_dict)

        data_dict['points'] = points

        if self.img_aug_type is not None:
            data_dict = self.copy_paste_to_image(img_aug_gt_dict, data_dict, points)

        return data_dict
    
    def frustum_double_mask_func(self, gt_dict, sampled):

        num_gt = gt_dict["gt_boxes"].shape[0]
        num_sampled = len(sampled)
    
        sp_frustums = np.stack([i["frustum"] for i in sampled], axis=0)
        gt_frustums = gt_dict["gt_frustums"]

        total_len = gt_frustums.shape[0] + sp_frustums.shape[0]

        frustum_coll_mat = self.frustum_collision_test(gt_frustums, sp_frustums)
        vein_coll_mat = frustum_coll_mat.zeros_like()

        coll_mat = np.logical_or(vein_coll_mat, frustum_coll_mat)

        diag = np.arange(total_len)
        coll_mat[diag, diag] = False

        valid_samples_indexes = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples_indexes.append(i - num_gt)

        return valid_samples_indexes


    def frustum_collision_test(self, gt_frustums, sp_frustums, thresh=0.7):
            ## calculate iou
            N = gt_frustums.shape[0]
            K = sp_frustums.shape[0]
            gt_frustums_all = np.concatenate([gt_frustums, sp_frustums], axis=0)
            S = np.array([(cur_frus[1, 1, 0] - cur_frus[1, 0, 0]) * (cur_frus[2, 1, 0] - cur_frus[2, 0, 0] + cur_frus[2, 1, 1] - cur_frus[2, 0, 1]) \
                        for cur_frus in gt_frustums_all], dtype=np.float32)
            # assert S.any() > 0
            ret = np.zeros((N+K, N+K), dtype=np.float32)
            for i in range(N+K):
                for j in range(K):
                    sp_frus = [sp_frustums[j, :, :, 0]] if sp_frustums[j, 2, 0, 1] < 0 else [sp_frustums[j, :, :, 0], sp_frustums[j, :, :, 1]]
                    gt_frus = [gt_frustums_all[i, :, :, 0]] if gt_frustums_all[i, 2, 0, 1] < 0 else [gt_frustums_all[i, :, :, 0], gt_frustums_all[i, :, :, 1]]
                    iou = 0
                    for cur_sp_frus in sp_frus:
                        for cur_gt_frus in gt_frus:
                            coll = ((max(cur_sp_frus[2, 0], cur_gt_frus[2, 0]) < min(cur_sp_frus[2, 1], cur_gt_frus[2, 1]))
                                and (max(sp_frustums[j, 1, 0, 0], gt_frustums_all[i, 1, 0, 0]) < min(sp_frustums[j, 1, 1, 0], gt_frustums_all[i, 1, 1, 0])))
                            if coll:
                                iou += (min(cur_sp_frus[2, 1], cur_gt_frus[2, 1]) - max(cur_sp_frus[2, 0], cur_gt_frus[2, 0])) * \
                                (min(sp_frustums[j, 1, 1, 0], gt_frustums_all[i, 1, 1, 0]) - max(sp_frustums[j, 1, 0, 0], gt_frustums_all[i, 1, 0, 0]))
                                # assert iou > 0

                    iou_per = iou / min(S[i], S[j+N])
                    # assert iou_per <= 1.01
                    ret[i, j + N] = iou_per
                    ret[j + N, i] = iou_per

            ret = ret > thresh
            return ret


    #FIXME: check这一块能否被PointAugmenting的GT Sampling完整替换
    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        sampled_mv_height = []
        sampled_gt_boxes2d = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            
            
            if int(sample_group['sample_num']) > 0:

                #TODO: 这里从GT Database里面根据需求直接采样，并不考虑冲突，在后面考虑冲突，和PointAugmenting不一样
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)


                # for k,v in sampled_dict:
                #     print(f"^^^K: {k}, V: {v}")
                
                # exit()

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                # sampled_points = np.stack([x['points'] for x in sampled_dict], axis=0).astype(np.float32)

                assert not self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False), 'Please use latest codes to generate GT_DATABASE'

                #TODO: 判断是否存在冲突
                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                #TODO: 将`iou2`对角线上的值设为0，避免计算自身边界框的IoU
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2

                #TODO: 根据计算得到的IoU结果，生成一个布尔掩码，用于判断采样边界框是否与已存在边界框存在冲突
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0)

                # img_aug_type = nuscenes
                if self.img_aug_type is not None:

                    #TODO: 并不是每一个sample box都会被选取，需要计算和当前frame的gt data dict的冲突情况
                    sampled_boxes2d, mv_height, valid_mask = self.sample_gt_boxes_2d(data_dict, sampled_boxes, valid_mask, sampled_dict)
                    sampled_gt_boxes2d.append(sampled_boxes2d)
                    if mv_height is not None:
                        sampled_mv_height.append(mv_height)

                valid_mask = valid_mask.nonzero()[0]
                if self.frustum_double_mask == True:
                    frustum_valid_indexes = self.frustum_double_mask_func(data_dict, sampled_dict)
                
                #FIXME: 同时在 valid_mask 和 frustum_valid_indexes 才加入
                valid_mask_ = list(set(valid_mask) & set(frustum_valid_indexes))


                torch.cuda.empty_cache()
                exit()

                valid_sampled_dict = [sampled_dict[x] for x in valid_mask_]
                valid_sampled_boxes = sampled_boxes[valid_mask_]


                
                #TODO: valid sampled boxes可以理解为“确认可以添加的目标”
                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes[:, :existed_boxes.shape[-1]]), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]

        sampled_num = sampled_gt_boxes.shape[0]
        data_dict["pasted"] = np.concatenate([np.zeros([data_dict["gt_boxes"].shape[0]]), np.ones(sampled_num)],
                    axis=0)

        if total_valid_sampled_dict.__len__() > 0:
            sampled_gt_boxes2d = np.concatenate(sampled_gt_boxes2d, axis=0) if len(sampled_gt_boxes2d) > 0 else None
            sampled_mv_height = np.concatenate(sampled_mv_height, axis=0) if len(sampled_mv_height) > 0 else None

            data_dict = self.add_sampled_boxes_to_scene(
                data_dict, sampled_gt_boxes, total_valid_sampled_dict, sampled_mv_height, sampled_gt_boxes2d
            )

        data_dict.pop('gt_boxes_mask')
        return data_dict
