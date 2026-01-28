# -*- coding: utf-8 -*-
import os
import subprocess

from omegaconf import DictConfig, OmegaConf

HF_COMMAND = "uv run hf download"


def check_all_checkpoints_exist(infer_cfg: DictConfig, model_checkpoints_path: str, audio_models: bool) -> bool:
    """
    check whether all checkpoints in a config exist. Method from FasterLivePortrait
    """
    ret = True
    for name in infer_cfg.models:
        if not isinstance(infer_cfg.models[name].model_path, str):
            for i in range(len(infer_cfg.models[name].model_path)):
                infer_cfg.models[name].model_path[i] = (
                    infer_cfg.models[name].model_path[i].replace("./checkpoints", model_checkpoints_path)
                )
                if not os.path.exists(infer_cfg.models[name].model_path[i]) and not os.path.exists(
                    infer_cfg.models[name].model_path[i][:-4] + ".onnx"
                ):
                    return False
        else:
            infer_cfg.models[name].model_path = infer_cfg.models[name].model_path.replace(
                "./checkpoints", model_checkpoints_path
            )
            if not os.path.exists(infer_cfg.models[name].model_path) and not os.path.exists(
                infer_cfg.models[name].model_path[:-4] + ".onnx"
            ):
                return False
    for name in infer_cfg.animal_models:
        if not isinstance(infer_cfg.animal_models[name].model_path, str):
            for i in range(len(infer_cfg.animal_models[name].model_path)):
                infer_cfg.animal_models[name].model_path[i] = (
                    infer_cfg.animal_models[name].model_path[i].replace("./checkpoints", model_checkpoints_path)
                )
                if not os.path.exists(infer_cfg.animal_models[name].model_path[i]) and not os.path.exists(
                    infer_cfg.animal_models[name].model_path[i][:-4] + ".onnx"
                ):
                    return False
        else:
            infer_cfg.animal_models[name].model_path = infer_cfg.animal_models[name].model_path.replace(
                "./checkpoints", model_checkpoints_path
            )
            if not os.path.exists(infer_cfg.animal_models[name].model_path) and not os.path.exists(
                infer_cfg.animal_models[name].model_path[:-4] + ".onnx"
            ):
                return False

    # XPOSE
    xpose_model_path = os.path.join(model_checkpoints_path, "liveportrait_animal_onnx/xpose.pth")
    if not os.path.exists(xpose_model_path):
        return False
    embeddings_cache_9_path = os.path.join(model_checkpoints_path, "liveportrait_animal_onnx/clip_embedding_9.pkl")
    if not os.path.exists(embeddings_cache_9_path):
        return False
    embeddings_cache_68_path = os.path.join(model_checkpoints_path, "liveportrait_animal_onnx/clip_embedding_68.pkl")
    if not os.path.exists(embeddings_cache_68_path):
        return False

    if audio_models:
        motion_generator_path = os.path.join(
            model_checkpoints_path, "JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
        )
        if not os.path.exists(motion_generator_path):
            return False
        audio_model_path = os.path.join(model_checkpoints_path, "chinese-hubert-base")
        if not os.path.exists(audio_model_path):
            return False
        motion_template_path = os.path.join(model_checkpoints_path, "JoyVASA/motion_template/motion_template.pkl")
        if not os.path.exists(motion_template_path):
            return False
    return ret


def run_download(command: str, path: str) -> None:
    print(f"download model: {command}")
    result = subprocess.run(command, shell=True, check=True)
    if result.returncode == 0:
        print(f"Download checkpoints to {path} successful")
    else:
        print(f"Download checkpoints to {path} failed")
        exit(1)


def download_models(model_checkpoints_path: str, audio_models: bool = False) -> None:
    """Download huggingface models for FasterLivePortrait at the given path

    Args:
        model_checkpoints_path (str): path to save models
    """

    download_cmd = f"{HF_COMMAND} warmshao/FasterLivePortrait --local-dir {model_checkpoints_path}"
    run_download(download_cmd, model_checkpoints_path)
    if audio_models:
        local_dir = os.path.join(model_checkpoints_path, "chinese-hubert-base")
        download_cmd = f"{HF_COMMAND} TencentGameMate/chinese-hubert-base --local-dir {local_dir}"
        run_download(download_cmd, model_checkpoints_path)

        local_dir = os.path.join(model_checkpoints_path, "JoyVASA")
        download_cmd = f"{HF_COMMAND} jdh-algo/JoyVASA --local-dir {local_dir}"
        run_download(download_cmd, model_checkpoints_path)


def check_models(cfg_file: str, model_checkpoints_path: str, audio_models: bool = False) -> bool:
    """Check if models exist at a given path

    Args:
        cfg_file (str): configuration file
        model_checkpoints_path (str): path to check models exist

    Returns:
        bool: _description_
    """
    infer_cfg = OmegaConf.load(cfg_file)
    checkpoints_exist = check_all_checkpoints_exist(infer_cfg, model_checkpoints_path, audio_models)
    return checkpoints_exist
