# -*- coding: utf-8 -*-
import datetime
import os
import pickle
import time
from typing import Any, Optional
import torch
import pathlib

import cv2
import numpy as np
from colorama import Fore, Style
from fasterliveportrait.pipelines.joyvasa_audio_to_motion_pipeline import (
    JoyVASAAudio2MotionPipeline,
)
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from fasterliveportrait.interface.pipeline import FasterLivePortraitPipelineOverride
from ..models.JoyVASA.dit_talking_head import DitTalkingHead
from ..models.JoyVASA.helper import NullableArgs
from ..utils import utils


class RunArgs(BaseModel):
    """Args for running pipeline"""

    cfg: str
    paste_back: bool
    animal: bool
    dri_video: str
    src_image: np.ndarray
    realtime: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class NullableArgsOverride(NullableArgs):
    def __init__(self, arg_dict):
        for key, value in arg_dict.items():
            setattr(self, key, value)


class JoyVASAAudio2MotionPipelineOverride(JoyVASAAudio2MotionPipeline):
    """
    JoyVASA 声音生成LivePortrait Motion
    """

    def __init__(self, **kwargs):
        self.device, self.dtype = utils.get_opt_device_dtype()
        # Check if the operating system is Windows
        if os.name == "nt":
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
        motion_model_path = kwargs.get("motion_model_path", "")
        audio_model_path = kwargs.get("audio_model_path", "")
        motion_template_path = kwargs.get("motion_template_path", "")
        model_data = torch.load(motion_model_path)
        model_args = NullableArgs(model_data["args"])
        model = DitTalkingHead(
            motion_feat_dim=model_args.motion_feat_dim,
            n_motions=model_args.n_motions,
            n_prev_motions=model_args.n_prev_motions,
            feature_dim=model_args.feature_dim,
            audio_model=model_args.audio_model,
            n_diff_steps=model_args.n_diff_steps,
            audio_encoder_path=audio_model_path,
        )
        model_data["model"].pop("denoising_net.TE.pe")
        model.load_state_dict(model_data["model"], strict=False)
        model.to(self.device, dtype=self.dtype)
        model.eval()

        # Restore the original PosixPath if it was changed
        if os.name == "nt":
            pathlib.PosixPath = temp

        self.motion_generator = model
        self.n_motions = model_args.n_motions
        self.n_prev_motions = model_args.n_prev_motions
        self.fps = model_args.fps
        self.audio_unit = 16000.0 / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = model_args.pad_mode
        self.use_indicator = model_args.use_indicator
        self.cfg_mode = kwargs.get("cfg_mode", "incremental")
        self.cfg_cond = kwargs.get("cfg_cond", None)
        self.cfg_scale = kwargs.get("cfg_scale", 2.8)
        with open(motion_template_path, "rb") as fin:
            self.templete_dict = pickle.load(fin)


def run_with_video(args: RunArgs) -> list[np.ndarray]:
    """Adapted method from FasterLivePortrait run.py
    Run pipeline with a video and return the generated frames.
    """
    frames: list[np.ndarray] = []
    print(
        Fore.RED
        + "Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo,"
        + "KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate"
        + Style.RESET_ALL
    )
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    pipe = FasterLivePortraitPipelineOverride(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)
    if not args.dri_video or not os.path.exists(args.dri_video):
        # read frame from camera if no driving video input
        vcap = cv2.VideoCapture(0)
        if not vcap.isOpened():
            print("no camera found! exit!")
            exit(1)
    else:
        vcap = cv2.VideoCapture(args.dri_video)

    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    infer_times = []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []

    frame_ind = 0
    pbar = tqdm(desc="Frames processed", unit="frames")
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_crop, out_crop, _, dri_motion_info = pipe.run(
            frame, pipe.src_imgs[0], pipe.src_infos[0], first_frame=first_frame
        )
        frame_ind += 1
        pbar.update(frame_ind)
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        motion_lst.append(dri_motion_info[0])
        c_eyes_lst.append(dri_motion_info[1])
        c_lip_lst.append(dri_motion_info[2])

        infer_times.append(time.time() - t0)
        dri_crop = cv2.resize(dri_crop, (512, 512))
        frames.append(out_crop)
    pbar.close()

    vcap.release()
    if args.realtime:
        cv2.destroyAllWindows()
    return frames


def run_with_pkl(
    args: RunArgs, dri_motion_sequence: Optional[dict[str, Any]] = None
) -> list[np.ndarray]:
    """Adapted method from FasterLivePortrait run.py
    Run pipeline with a motion file and return the generated frames.
    """
    frames: list[np.ndarray] = []
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    pipe = FasterLivePortraitPipelineOverride(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        return []
    if dri_motion_sequence is not None:
        dri_motion_infos = dri_motion_sequence
    else:
        with open(args.dri_video, "rb") as fin:
            dri_motion_infos = pickle.load(fin)

    infer_times = []
    motion_lst = dri_motion_infos["motion"]
    c_eyes_lst = (
        dri_motion_infos["c_eyes_lst"]
        if "c_eyes_lst" in dri_motion_infos
        else dri_motion_infos["c_d_eyes_lst"]
    )
    c_lip_lst = (
        dri_motion_infos["c_lip_lst"]
        if "c_lip_lst" in dri_motion_infos
        else dri_motion_infos["c_d_lip_lst"]
    )

    frame_num = len(motion_lst)
    for frame_ind in tqdm(range(frame_num)):
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_motion_info_ = [
            motion_lst[frame_ind],
            c_eyes_lst[frame_ind],
            c_lip_lst[frame_ind],
        ]
        out_crop, _ = pipe.run_with_pkl(
            dri_motion_info_,
            pipe.src_imgs[0],
            pipe.src_infos[0],
            first_frame=first_frame,
        )
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        infer_times.append(time.time() - t0)
        frames.append(out_crop)

    if args.realtime:
        cv2.destroyAllWindows()
    return frames


def run_audio_driving(driving_audio: bytes, args: RunArgs) -> list[np.ndarray]:
    infer_cfg = OmegaConf.load(args.cfg)

    joyvasa_pipe = JoyVASAAudio2MotionPipeline(
        motion_model_path=infer_cfg.joyvasa_models.motion_model_path,
        audio_model_path=infer_cfg.joyvasa_models.audio_model_path,
        motion_template_path=infer_cfg.joyvasa_models.motion_template_path,
        cfg_mode=infer_cfg.infer_params.cfg_mode,
        cfg_scale=infer_cfg.infer_params.cfg_scale,
    )
    dri_motion_infos = joyvasa_pipe.gen_motion_sequence(driving_audio)
    frames = run_with_pkl(args, dri_motion_infos)
    return frames


def run_pipeline(args: RunArgs, audio: Optional[bytes] = None) -> list[np.ndarray]:
    if audio is not None:
        frames = run_audio_driving(audio, args)
    else:
        frames = (
            run_with_pkl(args=args)
            if args.dri_video.endswith(".pkl")
            else run_with_video(args=args)
        )
    return frames
