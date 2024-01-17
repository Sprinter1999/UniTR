import numpy as np
import cv2
from .cross_modal_augmentation import *
import pickle
from .sample_ops import DataBaseSamplerV2
# import preprocess as prep



def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds




def build_dbsampler(cfg, logger=None):
    # logger = logging.getLogger("build_dbsampler")
    prepors = [build_db_preprocess(c, logger=logger) for c in cfg.db_prep_steps]
    db_prepor = DataBasePreprocessor(prepors)
    rate = cfg.rate
    grot_range = cfg.global_random_rotation_range_per_object
    groups = cfg.sample_groups
    # groups = [dict(g.name_to_max_num) for g in groups]
    info_path = cfg.db_info_path
    with open(info_path, "rb") as f:
        db_infos = pickle.load(f)
    grot_range = list(grot_range)
    if len(grot_range) == 0:
        grot_range = None
    sampler = DataBaseSamplerV2(
        db_infos, groups, db_prepor, rate, grot_range, logger=logger
    )

    return sampler

class DataBaseSampler_PA(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = True
        self.min_points_in_gt = 5
        
        self.mode = cfg.mode
        if self.mode == "train":
            # self.global_rotation_noise = cfg.global_rot_noise
            # self.global_scaling_noise = cfg.global_scale_noise
            # self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            self.remove_points_after_sample = cfg.get('remove_points_after_sample', False)
            # if cfg.db_sampler != None:
            self.db_sampler = build_dbsampler(cfg.db_sampler)
            # else:
            #     self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

        self.use_img = True


    #FIXME: check这一块能否接入UniTR的代码库
    def __call__(self, data_dict):

        # res["mode"] = self.mode

        self.mode = 'train'

        points = data_dict['points']
        # if res["type"] in ["WaymoDataset"]:
        #     if "combined" in res["lidar"]:
        #         points = res["lidar"]["combined"]
        #     else:
        #         points = res["lidar"]["points"]
        # elif res["type"] in ["NuScenesDataset"]:
        #     points = res["lidar"]["combined"]
        # else:
        #     raise NotImplementedError

        cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


        if self.mode == "train":
            # anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": data_dict['gt_boxes'],
                "gt_names": np.array(data_dict["names"]).reshape(-1),
                "gt_frustums": data_dict["frustums"],
            }

            cam_anno_dict = {
                    "avail_2d": data_dict["avail_2d"].astype(np.bool),
                    "boxes_2d": data_dict["boxes_2d"].astype(np.int32),
                    "depths": data_dict["depths"].astype(np.float32),
            }

            if self.use_img:
                # cam_anno_dict = res["camera"]["annotations"]
                gt_dict["bboxes"] = cam_anno_dict["boxes_2d"]
                gt_dict["avail_2d"] = cam_anno_dict["avail_2d"]
                gt_dict["depths"] = cam_anno_dict["depths"]

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            # if self.db_sampler:
            # calib = res["calib"] if "calib" in res else None
            calib = data_dict["calib"]
            selected_feature = np.ones([5 + 3])  # xyzrt, u v cam_id
            selected_feature[5:5 + 3] = 1. if self.use_img else 0.

            #TODO: 直接return了被采样的目标们，在这里sample的时候，将当前帧传进来了，同时考虑冲突的问题
            sampled_dict = self.db_sampler.sample_all_v2(
                data_dict["image_paths"], #res["metadata"]["image_prefix"],
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                gt_dict["gt_frustums"],
                selected_feature,
                random_crop=False,
                revise_calib=True,
                gt_group_ids=None,
                calib=calib,
                cam_name=cam_names, #res['camera']['name'],
                road_planes=None  # res["lidar"]["ground_plane"]
            )

            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                sampled_frustums = sampled_dict["gt_frustums"]
                sampled_num = sampled_gt_boxes.shape[0]
                origin_num = gt_dict['gt_boxes'].shape[0]
                gt_dict["pasted"] = np.concatenate([np.zeros([gt_dict["gt_boxes"].shape[0]]), np.ones(sampled_num)],
                                                    axis=0)
                if self.use_img:
                    gt_dict["avail_2d"] = np.concatenate([gt_dict["avail_2d"], sampled_dict["avail_2d"]], axis=0)
                    gt_dict["bboxes"] = np.concatenate([gt_dict["bboxes"], sampled_dict["bboxes"]], axis=0)
                    gt_dict["depths"] = np.concatenate([gt_dict["depths"], sampled_dict["depths"]], axis=0)
                    gt_dict["patch_path"] = np.concatenate(
                        [[['']*6 for i in range(origin_num)], sampled_dict["patch_path"]], axis=0)


                #TODO: 本质上，将 sampled_gt_boxes 的信息装饰到gt_dict
                    
                # gt_boxes放最后
                gt_dict["gt_names"] = np.concatenate([gt_dict["gt_names"], sampled_gt_names], axis=0)
                gt_dict["gt_boxes"] = np.concatenate([gt_dict["gt_boxes"], sampled_gt_boxes])
                gt_dict["gt_frustums"] = np.concatenate([gt_dict["gt_frustums"], sampled_frustums])
                gt_boxes_mask = np.concatenate([gt_boxes_mask, sampled_gt_masks], axis=0)

                if self.remove_points_after_sample:
                    masks = points_in_rbbox(points, sampled_gt_boxes)
                    points = points[np.logical_not(masks.any(-1))]

                #TODO: PointAugmenting
                if self.use_img:  # paste imgs
                    # res['img'] = procress_image(res['img'], gt_dict)
                    data_dict['images'] = procress_image(data_dict['images'], gt_dict)

                # from tools.visualization import show_pts_in_box
                # show_pts_in_box(points, sampled_points)

                #TODO: PointAugmenting
                points, gt_boxes_mask = procress_points(points, sampled_points, gt_boxes_mask, gt_dict)

            if self.use_img:
                gt_dict.pop('avail_2d')
                gt_dict.pop('bboxes')
                gt_dict.pop('depths')

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


            #TODO: 这里的后续增强就留到UniTR框架本身自带的增强Pipeline了, 不在這裏處理
            # gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            # gt_dict["gt_boxes"], points = prep.global_rotation(
            #     gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            # )
            # gt_dict["gt_boxes"], points = prep.global_scaling_v2(
            #     gt_dict["gt_boxes"], points, *self.global_scaling_noise
            # )
            # gt_dict["gt_boxes"], points = prep.global_translate_(
            #     gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
            # )


        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

        # if self.shuffle_points:
        np.random.shuffle(points)

        if self.use_img:
            points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1).astype(np.float32)
        # res["lidar"]["points"] = points
        
        data_dict["points"] = points
                  
        if self.mode == "train":
            # res["lidar"]["annotations"] = gt_dict
            data_dict["gt_boxes"] = gt_dict["gt_boxes"] 
            data_dict["gt_names"] = gt_dict["gt_names"]
            data_dict["gt_frustums"] = gt_dict["gt_frustums"]

        return data_dict
