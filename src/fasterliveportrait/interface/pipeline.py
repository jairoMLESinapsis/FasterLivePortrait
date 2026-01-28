# -*- coding: utf-8 -*-
import copy
import traceback
from typing import Any

import cv2
import numpy as np
import torch
from fasterliveportrait.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from fasterliveportrait.utils.crop import crop_image
from fasterliveportrait.utils.utils import (
    concat_feat,
    get_rotation_matrix,
    prepare_paste_back,
    resize_to_limit,
    transform_keypoint,
)
from PIL import Image
from tqdm import tqdm


class FasterLivePortraitPipelineOverride(FasterLivePortraitPipeline):
    """Override to accept imaages as input instead of filepath. Adapted from FasterLivePortrait"""

    def prepare_source(self, img_bgr: np.ndarray, **kwargs: Any) -> bool:
        try:
            src_imgs_bgr = [img_bgr]

            self.src_imgs = []
            self.src_infos = []
            # self.source_path = source_path

            for ii, img_bgr in tqdm(enumerate(src_imgs_bgr), total=len(src_imgs_bgr)):
                img_bgr = resize_to_limit(
                    img_bgr, self.cfg.infer_params.source_max_dim, self.cfg.infer_params.source_division
                )
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                src_faces = []
                if self.is_animal:
                    with torch.no_grad():
                        img_rgb_pil = Image.fromarray(img_rgb)
                        lmk = self.model_dict["xpose"].run(img_rgb_pil, "face", "animal_face", 0, 0)
                    if lmk is None:
                        continue
                    self.src_imgs.append(img_rgb)
                    src_faces.append(lmk)
                else:
                    src_faces = self.model_dict["face_analysis"].predict(img_bgr)
                    if len(src_faces) == 0:
                        print("No face detected in the this image.")
                        continue
                    self.src_imgs.append(img_rgb)
                    # 如果是实时 只关注最大的那张脸
                    if kwargs.get("realtime", False):
                        src_faces = src_faces[:1]

                crop_infos = []
                for i in range(len(src_faces)):
                    # NOTE: temporarily only pick the first face, to support multiple face in the future
                    lmk = src_faces[i]
                    # crop the face
                    ret_dct = crop_image(
                        img_rgb,  # ndarray
                        lmk,  # 106x2 or Nx2
                        dsize=self.cfg.crop_params.src_dsize,
                        scale=self.cfg.crop_params.src_scale,
                        vx_ratio=self.cfg.crop_params.src_vx_ratio,
                        vy_ratio=self.cfg.crop_params.src_vy_ratio,
                    )
                    if self.is_animal:
                        ret_dct["lmk_crop"] = lmk
                    else:
                        lmk = self.model_dict["landmark"].predict(img_rgb, lmk)
                        ret_dct["lmk_crop"] = lmk
                        ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / self.cfg.crop_params.src_dsize

                    # update a 256x256 version for network input
                    ret_dct["img_crop_256x256"] = cv2.resize(
                        ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA
                    )
                    crop_infos.append(ret_dct)

                src_infos: list[list[Any]] = [[] for _ in range(len(crop_infos))]
                for i, crop_info in enumerate(crop_infos):
                    source_lmk = crop_info["lmk_crop"]
                    _, img_crop_256x256 = crop_info["img_crop"], crop_info["img_crop_256x256"]
                    pitch, yaw, roll, t, exp, scale, kp = self.model_dict["motion_extractor"].predict(img_crop_256x256)
                    x_s_info = {"pitch": pitch, "yaw": yaw, "roll": roll, "t": t, "exp": exp, "scale": scale, "kp": kp}
                    src_infos[i].append(copy.deepcopy(x_s_info))
                    x_c_s = kp
                    R_s = get_rotation_matrix(pitch, yaw, roll)
                    f_s = self.model_dict["app_feat_extractor"].predict(img_crop_256x256)
                    x_s = transform_keypoint(pitch, yaw, roll, t, exp, scale, kp)
                    src_infos[i].extend([source_lmk.copy(), R_s.copy(), f_s.copy(), x_s.copy(), x_c_s.copy()])
                    if not self.is_animal:
                        flag_lip_zero = self.cfg.infer_params.flag_normalize_lip  # not overwrite
                        if flag_lip_zero:
                            # let lip-open scalar to be 0 at first
                            # 似乎要调参
                            c_d_lip_before_animation = [0.05]
                            combined_lip_ratio_tensor_before_animation = self.calc_combined_lip_ratio(
                                c_d_lip_before_animation, source_lmk.copy()
                            )
                            if (
                                combined_lip_ratio_tensor_before_animation[0][0]
                                < self.cfg.infer_params.lip_normalize_threshold
                            ):
                                flag_lip_zero = False
                                src_infos[i].append(None)
                                src_infos[i].append(flag_lip_zero)
                            else:
                                lip_delta_before_animation = self.model_dict["stitching_lip_retarget"].predict(
                                    concat_feat(x_s, combined_lip_ratio_tensor_before_animation)
                                )
                                src_infos[i].append(lip_delta_before_animation.copy())
                                src_infos[i].append(flag_lip_zero)
                        else:
                            src_infos[i].append(None)
                            src_infos[i].append(flag_lip_zero)
                    else:
                        src_infos[i].append(None)
                        src_infos[i].append(False)

                    ######## prepare for pasteback ########
                    if (
                        self.cfg.infer_params.flag_pasteback
                        and self.cfg.infer_params.flag_do_crop
                        and self.cfg.infer_params.flag_stitching
                    ):
                        mask_ori_float = prepare_paste_back(
                            self.mask_crop, crop_info["M_c2o"], dsize=(img_rgb.shape[1], img_rgb.shape[0])
                        )
                        mask_ori_float = torch.from_numpy(mask_ori_float).to(self.device)
                        src_infos[i].append(mask_ori_float)
                    else:
                        src_infos[i].append(None)
                    M = torch.from_numpy(crop_info["M_c2o"]).to(self.device)
                    src_infos[i].append(M)
                self.src_infos.append(src_infos[:])
            return len(self.src_infos) > 0
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return False
