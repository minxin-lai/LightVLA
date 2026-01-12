"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
import multiprocessing
from pathlib import Path
import time
import uuid
from typing import Optional, Union, Any, Dict, List

import draccus
import imageio
from joblib import Parallel, delayed
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.openvla_utils import model_is_on_hf_hub, update_auto_map, check_model_logic_mismatch, _apply_film_to_vla, _load_dataset_stats

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    prepare_images_for_vla,
    normalize_proprio,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from scripts.trace.pruner_trace import TokenPrunerTracer, to_cpu
from scripts.trace.schema import SCHEMA_VERSION


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./val_logs"                # Local directory for eval logs
    save_rollout_video: bool = False                 # Save rollout videos

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # Tracing (Step 5/6): dump routing + save images for overlay
    #################################################################################################################
    trace_out_dir: str = ""                           # If set, write dumps/images under this directory
    trace_max_dumps_per_run: int = 500                # Safety cap to avoid excessive disk usage
    trace_save_policy_images: bool = True             # Save policy input images (224x224) for overlay
    trace_dump_routing: bool = True                   # Dump routing signals from TokenPruner
    trace_dump_attn: bool = False                     # Dump reduced LLM self-attention evidence (task→vision)
    trace_attn_layers: str = "31"                     # Comma-separated layer indices, e.g. "0,31"

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def get_gpu_memory_usage():
    """Get GPU memory usage information."""
    
    n_gpus = torch.cuda.device_count()

    gpu_info = []

    for gpu_id in range(n_gpus):
        try:
            mem_avail = torch.cuda.mem_get_info(gpu_id)[0]
        except:
            mem_avail = 0

        gpu_info.append({
            'id': gpu_id,
            'memory_available': mem_avail / 1e6,
        })
    
    # Sort by memory usage (descending)
    gpu_info.sort(key=lambda x: x['memory_available'], reverse=True)

    return gpu_info


def get_model(cfg, device) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    # If loading a locally stored pretrained checkpoint, check whether config or model files
    # need to be synced so that any changes the user makes to the VLA modeling code will
    # actually go into effect
    # If loading a pretrained checkpoint from Hugging Face Hub, we just assume that the policy
    # will be used as is, with its original modeling logic
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    # Load the model
    vla = OpenVLAForActionPrediction.from_pretrained(
        cfg.pretrained_checkpoint,
        device_map=device,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # Load dataset stats for action normalization
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def _trace_paths(cfg: GenerateConfig, task_id: int, episode_idx: int, t: int, query_idx: int) -> Dict[str, Any]:
    """
    Create unique file paths for this trace record.
    """
    run_root = Path(cfg.trace_out_dir)
    # Organize by task/episode to avoid huge flat directories.
    dumps_dir = run_root / "dumps" / f"task{task_id:03d}" / f"ep{episode_idx:03d}"
    images_dir = run_root / "images" / f"task{task_id:03d}" / f"ep{episode_idx:03d}"
    report_dir = run_root / "report"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:10]
    sample_id = f"task{task_id:03d}_ep{episode_idx:03d}_t{t:04d}_q{query_idx:03d}_{os.getpid()}_{uid}"
    dump_path = dumps_dir / f"{sample_id}.pt"
    return {
        "run_root": str(run_root),
        "sample_id": sample_id,
        "dump_path": str(dump_path),
        "images_dir": str(images_dir),
        "report_dir": str(report_dir),
    }


def _get_patch_grid_hw(vla: torch.nn.Module) -> Optional[tuple]:
    try:
        grid = vla.vision_backbone.featurizer.patch_embed.grid_size
        if isinstance(grid, tuple) and len(grid) == 2:
            return int(grid[0]), int(grid[1])
    except Exception:
        pass
    return None


def _parse_int_list(csv: str) -> List[int]:
    out: List[int] = []
    for part in (csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def initialize_model(cfg: GenerateConfig, device):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg, device)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        ).to(device)

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim).to(device)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim).to(device)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    log_file = os.path.join(cfg.local_log_dir, run_id + ".txt")
    logger.info(f"Logging to local log file: {log_file}")

    return log_file, run_id


def log_message(message: str, log_file=None, log_lock=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        with log_lock:
            with open(log_file, 'a') as f:
                f.write(message + "\n")
                f.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None, log_lock=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file, log_lock)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file, log_lock)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def save_rollout_video(rollout_images, task_id, idx, success, task_description):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--openvla_oft--task={task_id}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    return mp4_path


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    device: Any = None,
    trace_ctx: Optional[Dict[str, Any]] = None,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)

        # Build VLA prompt
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process primary image
        inputs = processor(prompt, primary_image).to(device, dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(device, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            # Concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        # Optional tracing: capture TokenPruner routing signals + save policy input images for overlay.
        do_trace = bool(cfg.trace_out_dir) and trace_ctx is not None and (cfg.trace_max_dumps_per_run > 0)
        if do_trace and trace_ctx.get("dump_count", 0) >= cfg.trace_max_dumps_per_run:
            do_trace = False

        tracer = None
        pruner = getattr(getattr(vla, "language_model", None), "model", None)
        pruner = getattr(pruner, "pruner", None)  # language_model.model.pruner
        if do_trace and cfg.trace_dump_routing and pruner is not None:
            tracer = TokenPrunerTracer(pruner, store_raw_score=False)

        # Optional LLM attention capture: reduce to a compact task→vision vector per selected layer.
        attn_task_to_vis_by_layer: Dict[str, torch.Tensor] = {}
        attn_handles = []
        prev_output_attentions = None
        if do_trace and cfg.trace_dump_attn:
            try:
                layer_idxs = _parse_int_list(cfg.trace_attn_layers)
            except Exception:
                layer_idxs = []

            llama_model = getattr(getattr(vla, "language_model", None), "model", None)
            layers = getattr(llama_model, "layers", None)
            if layers is not None and layer_idxs:
                # Force output_attentions to True so SDPA attention can fall back to manual attention with weights.
                prev_output_attentions = getattr(llama_model.config, "output_attentions", None)
                llama_model.config.output_attentions = True

                def make_hook(layer_idx: int):
                    def hook_fn(_module, _inp, out):
                        try:
                            if tracer is None:
                                return
                            keep_mask = tracer.trace.keep_mask
                            if keep_mask is None:
                                return
                            # out: (attn_output, attn_weights, past_key_value)
                            if not isinstance(out, (tuple, list)) or len(out) < 2:
                                return
                            attn_weights = out[1]
                            if attn_weights is None:
                                return
                            # attn_weights: [B, H, Q, K]
                            attn_mean = attn_weights.mean(dim=1)  # [B, Q, K]

                            # Pruned sequence layout: [cls(1), kept_vision(num_kept), task(rest)]
                            keep_mask_b = keep_mask[0].detach()
                            kept_idx = torch.nonzero(keep_mask_b, as_tuple=False).squeeze(-1)  # [num_kept]
                            num_kept = int(kept_idx.numel())
                            if num_kept <= 0:
                                return

                            q_len = int(attn_mean.shape[1])
                            k_len = int(attn_mean.shape[2])
                            if q_len != k_len:
                                # Should be self-attention, but guard anyway.
                                return

                            q_start = 1 + num_kept
                            k_start, k_end = 1, 1 + num_kept
                            if q_start >= q_len:
                                return

                            task_to_vis = attn_mean[:, q_start:, k_start:k_end].mean(dim=1)  # [B, num_kept]

                            # Scatter back to original patch index space [B, P_total]
                            p_total = int(keep_mask.shape[1])
                            full = torch.zeros(task_to_vis.shape[0], p_total, dtype=task_to_vis.dtype, device=task_to_vis.device)
                            full[:, kept_idx] = task_to_vis
                            attn_task_to_vis_by_layer[str(layer_idx)] = full.detach().to("cpu")
                        except Exception:
                            return

                    return hook_fn

                for li in layer_idxs:
                    if li < 0 or li >= len(layers):
                        continue
                    handle = layers[li].self_attn.register_forward_hook(make_hook(li))
                    attn_handles.append(handle)

        t0 = time.time()
        if tracer is not None:
            tracer.__enter__()
        try:
            # Generate action
            if action_head is None:
                # Standard VLA output (single-image inputs, discrete actions)
                action, _ = vla.predict_action(
                    **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, output_attentions=cfg.trace_dump_attn
                )
            else:
                # Custom action head for continuous actions
                action, _ = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    output_attentions=cfg.trace_dump_attn,
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    use_film=use_film,
                )
        finally:
            if tracer is not None:
                tracer.__exit__(None, None, None)
            for h in attn_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            if prev_output_attentions is not None:
                try:
                    llama_model = getattr(getattr(vla, "language_model", None), "model", None)
                    if llama_model is not None:
                        llama_model.config.output_attentions = prev_output_attentions
                except Exception:
                    pass
        latency_s = time.time() - t0

        if do_trace and cfg.trace_save_policy_images:
            try:
                trace_ctx["policy_images"] = [primary_image] + list(all_images)
            except Exception:
                trace_ctx["policy_images"] = [primary_image]

        if do_trace and tracer is not None:
            trace = tracer.trace
            if trace.indices is not None and trace.keep_mask is not None and trace.keep_counts is not None:
                # Align/routing signals are per-vision-token (patch) and depend on `num_images_in_input`.
                keep_mask_cpu = to_cpu(trace.keep_mask)
                indices_cpu = to_cpu(trace.indices)
                keep_counts_cpu = to_cpu(trace.keep_counts)

                num_patches_total = int(getattr(pruner, "num_patches", keep_mask_cpu.shape[1]))
                num_kept = keep_mask_cpu.sum(dim=-1).to(torch.long)
                kept_ratio = num_kept.to(torch.float32) / float(num_patches_total)
                router_scores = keep_counts_cpu.to(torch.float32) / float(num_patches_total)

                patch_grid_hw = _get_patch_grid_hw(vla)
                num_images_in_input = int(getattr(vla.vision_backbone, "num_images_in_input", 1))
                num_patches_per_image = num_patches_total // max(num_images_in_input, 1)

                image_paths: List[str] = []
                if cfg.trace_save_policy_images and "policy_images" in trace_ctx:
                    for img_idx, pil_img in enumerate(trace_ctx["policy_images"]):
                        img_path = Path(trace_ctx["images_dir"]) / f"{trace_ctx['sample_id']}__img{img_idx}.png"
                        pil_img.save(img_path)
                        image_paths.append(str(img_path))

                dump = {
                    "schema_version": SCHEMA_VERSION,
                    "meta": {
                        "sample_id": trace_ctx["sample_id"],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "model_id": str(Path(cfg.pretrained_checkpoint).name),
                        "instruction": task_label,
                        "task_suite": cfg.task_suite_name,
                        "task_id": trace_ctx.get("task_id"),
                        "episode_idx": trace_ctx.get("episode_idx"),
                        "step_idx": trace_ctx.get("t"),
                        "checkpoint_path": cfg.pretrained_checkpoint,
                    },
                    "align": {
                        "image_size_hw": [(img.height, img.width) for img in trace_ctx.get("policy_images", [])],
                        "image_paths": image_paths,
                        "patch_grid_hw": patch_grid_hw,
                        "num_images_in_input": num_images_in_input,
                        "num_patches_per_image": num_patches_per_image,
                        "num_patches_total": num_patches_total,
                        "seq_layout": {
                            "cls_range": (0, 1),
                            "vision_range": (1, 1 + num_patches_total),
                            "task_range": (1 + num_patches_total, None),
                        },
                    },
                    "routing": {
                        "num_patches": num_patches_total,
                        "selection_mode": "inference_hard_mask",
                        "router_scores": router_scores,
                        "indices": indices_cpu,
                        "keep_mask": keep_mask_cpu,
                        "keep_counts": keep_counts_cpu,
                        "num_kept": num_kept,
                        "kept_ratio": kept_ratio,
                        "seq_lens": {
                            "pre_seq_len": trace.pre_seq_len,
                            "post_seq_len": trace.post_seq_len,
                            "pre_task_len": trace.pre_task_len,
                            "post_task_len": trace.post_task_len,
                        },
                        "score_stats": {
                            "max_per_query": to_cpu(trace.score_max_per_query),
                            "entropy_per_query": to_cpu(trace.score_entropy_per_query),
                        },
                    },
                    "attn": {
                        "task_to_vis": attn_task_to_vis_by_layer,
                        "layers": sorted([int(k) for k in attn_task_to_vis_by_layer.keys()]) if attn_task_to_vis_by_layer else [],
                        "reduce": "head_mean + task_query_mean",
                    },
                    "perf": {"forward_latency_s": float(latency_s)},
                    "capabilities": [
                        "routing_scores",
                        "routing_indices",
                        "routing_keep_mask",
                        "routing_keep_counts",
                        "patch_grid_hw",
                        "token_counts",
                        "routing_score_max_per_query",
                        "routing_score_entropy_per_query",
                    ],
                }
                if attn_task_to_vis_by_layer:
                    dump["capabilities"].append("attn_task_to_vis")
                torch.save(dump, trace_ctx["dump_path"])
                trace_ctx["dump_count"] = int(trace_ctx.get("dump_count", 0)) + 1

    # Return action chunk as list of actions
    return [action[i] for i in range(len(action))]


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    task_id: int,
    episode_idx: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    log_lock=None,
    device=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    query_idx = 0
    dump_count = 0
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                trace_ctx = None
                if cfg.trace_out_dir:
                    trace_ctx = _trace_paths(cfg, task_id=task_id, episode_idx=episode_idx, t=t, query_idx=query_idx)
                    trace_ctx.update({"task_id": task_id, "episode_idx": episode_idx, "t": t})
                    trace_ctx["dump_count"] = dump_count

                # Query model to get action
                actions = get_vla_action(
                    cfg,
                    model,
                    processor,
                    observation,
                    task_description,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                    cfg.use_film,
                    device,
                    trace_ctx=trace_ctx,
                )
                action_queue.extend(actions)
                query_idx += 1
                if trace_ctx is not None:
                    dump_count = int(trace_ctx.get("dump_count", dump_count))

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file, log_lock=log_lock)

    return success, replay_images


def run_task(
    model_lock,
    log_lock,
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    resize_size,
    log_file=None,
    device=None,
):
    """Run evaluation for a single task."""

    # Set random seed
    set_seed_everywhere(cfg.seed+task_id)

    with model_lock:
        while True:
            gpu_info = get_gpu_memory_usage()
            if gpu_info[0]['memory_available'] > 20000:
                device = torch.device(f'cuda:{gpu_info[0]["id"]}')
                log_message(f'Task {task_id}: Selected GPU {gpu_info[0]["id"]}', log_file, log_lock)
                break
            log_message(f"Task {task_id}: Waiting for GPU available...", log_file, log_lock)
            time.sleep(60)

        # Initialize model and components
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg, device)

    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file, log_lock)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in range(cfg.num_trials_per_task):
        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file, log_lock)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Task {task_id}: Starting episode {task_episodes + 1}...", log_file, log_lock)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            task_id,
            episode_idx,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            log_lock,
            device,
        )

        # Update counters
        task_episodes += 1
        if success:
            task_successes += 1

        # Save replay video
        if cfg.save_rollout_video or not success:
            save_rollout_video(
                replay_images, task_id, task_episodes, success=success, task_description=task_description
            )

        # Log results
        log_message(f"\nTask {task_id}: {task_description}", log_file, log_lock)
        log_message(f"Task {task_id}: Success: {success}", log_file, log_lock)
        log_message(f"Task {task_id}: # episodes completed so far: {task_episodes}", log_file, log_lock)
        log_message(f"Task {task_id}: # successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)", log_file, log_lock)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0

    log_message(f"Task {task_id}: success rate: {task_success_rate}", log_file, log_lock)

    del model, action_head, proprio_projector, noisy_action_projector, processor
    torch.cuda.empty_cache()

    return task_successes, task_episodes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    manager = multiprocessing.Manager()
    model_lock = manager.Lock()
    log_lock = manager.Lock()

    log_message(f"Task suite: {cfg.task_suite_name}", log_file, log_lock)

    # Start evaluation
    results = Parallel(n_jobs=num_tasks)(delayed(run_task)(model_lock, log_lock, cfg, task_suite, task_id, resize_size, log_file) for task_id in range(num_tasks))
    
    total_successes = sum(task[0] for task in results)
    total_episodes = sum(task[1] for task in results)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file, log_lock)
    log_message(f"Total episodes: {total_episodes}", log_file, log_lock)
    log_message(f"Total successes: {total_successes}", log_file, log_lock)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file, log_lock)

    if cfg.trace_out_dir:
        try:
            from scripts.trace.run_report import write_capabilities_json

            out_path, _ = write_capabilities_json(
                Path(cfg.trace_out_dir),
                extra={
                    "task_suite": cfg.task_suite_name,
                    "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
                    "final_success_rate": float(final_success_rate),
                    "trace_dump_routing": bool(cfg.trace_dump_routing),
                    "trace_dump_attn": bool(cfg.trace_dump_attn),
                    "trace_attn_layers": str(cfg.trace_attn_layers),
                },
            )
            log_message(f"Trace report written: {out_path}", log_file, log_lock)
        except Exception as e:
            log_message(f"WARNING: failed to write trace report: {e}", log_file, log_lock)

        try:
            from scripts.trace.tokens_log import write_tokens_log

            out_jsonl, out_summary, _ = write_tokens_log(Path(cfg.trace_out_dir))
            log_message(f"Token log written: {out_jsonl}", log_file, log_lock)
            log_message(f"Token summary written: {out_summary}", log_file, log_lock)
        except Exception as e:
            log_message(f"WARNING: failed to write token log: {e}", log_file, log_lock)

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
