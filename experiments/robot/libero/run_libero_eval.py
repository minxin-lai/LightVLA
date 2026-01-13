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
from typing import Optional, Union, Any, Dict, List

import draccus
import imageio
from joblib import Parallel, delayed
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark

# Ensure local LightVLA packages (e.g., `experiments/`, `prismatic/`) take precedence over parent repo packages
# when `PYTHONPATH` includes the parent repo root (needed for `tracer`).
_LIGHTVLA_ROOT = Path(__file__).resolve().parents[3]
if str(_LIGHTVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIGHTVLA_ROOT))

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

try:
    from tracer.adapters.lightvla import LightVLATraceConfig, LightVLATracer
    from tracer.run_writer import TraceSample, TraceWriter
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing `tracer` module. Ensure both LightVLA and parent repo are on PYTHONPATH, e.g.:\n"
        "  export PYTHONPATH=/workspace/laiminxin/vla-opt/third_party/LightVLA:/workspace/laiminxin/vla-opt:$PYTHONPATH\n"
        "  python experiments/robot/libero/run_libero_eval.py ...\n"
    ) from e


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
    trace_max_dumps_per_run: int = 1                  # Safety cap to avoid excessive disk usage (0 = unlimited)
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
    trace_writer: Optional[TraceWriter] = None,
    trace_sample: Optional[TraceSample] = None,
    trace_task_id: Optional[int] = None,
    trace_episode_idx: Optional[int] = None,
    trace_step_idx: Optional[int] = None,
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

        do_trace = (
            bool(cfg.trace_out_dir)
            and trace_writer is not None
            and trace_sample is not None
            and bool(trace_sample.should_dump)
        )

        attn_layers = tuple(int(p.strip()) for p in str(cfg.trace_attn_layers).split(",") if p.strip())
        trace_cfg = LightVLATraceConfig(
            dump_routing=bool(cfg.trace_dump_routing),
            dump_attn=bool(cfg.trace_dump_attn),
            attn_layers=attn_layers,
            store_raw_score=False,
            save_policy_images=bool(cfg.trace_save_policy_images),
            max_dumps_per_run=int(cfg.trace_max_dumps_per_run),
        )

        tracer = LightVLATracer(vla, config=trace_cfg) if do_trace else None

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
        latency_s = time.time() - t0

        if do_trace and tracer is not None and trace_writer is not None and trace_sample is not None:
            policy_images = [primary_image] + list(all_images) if cfg.trace_save_policy_images else None
            dump = tracer.build_dump(
                meta={
                    "sample_id": trace_sample.sample_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model_id": str(Path(cfg.pretrained_checkpoint).name),
                    "instruction": task_label,
                    "task_suite": cfg.task_suite_name,
                    "task_id": trace_task_id,
                    "episode_idx": trace_episode_idx,
                    "step_idx": trace_step_idx,
                    "checkpoint_path": str(cfg.pretrained_checkpoint),
                },
                policy_images=policy_images,
                images_dir=str(trace_sample.images_dir),
                dump_path=str(trace_sample.dump_path),
                perf={"forward_latency_s": float(latency_s)},
            )
            if dump is not None:
                trace_writer.write_dump(dump, dump_path=trace_sample.dump_path)

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
    trace_writer = TraceWriter(Path(cfg.trace_out_dir), max_dumps_per_run=int(cfg.trace_max_dumps_per_run)) if cfg.trace_out_dir else None
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
                trace_sample = None
                if trace_writer is not None:
                    trace_sample = trace_writer.new_sample(
                        task_id=task_id,
                        episode_idx=episode_idx,
                        step_idx=t,
                        query_idx=query_idx,
                        layout="task_ep",
                    )

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
                    trace_writer=trace_writer,
                    trace_sample=trace_sample,
                    trace_task_id=task_id,
                    trace_episode_idx=episode_idx,
                    trace_step_idx=t,
                )
                action_queue.extend(actions)
                query_idx += 1

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
            writer = TraceWriter(Path(cfg.trace_out_dir))
            out_path, out_jsonl = writer.finalize(
                extra={
                    "task_suite": cfg.task_suite_name,
                    "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
                    "final_success_rate": float(final_success_rate),
                    "trace_dump_routing": bool(cfg.trace_dump_routing),
                    "trace_dump_attn": bool(cfg.trace_dump_attn),
                    "trace_attn_layers": str(cfg.trace_attn_layers),
                }
            )
            log_message(f"Trace report written: {out_path}", log_file, log_lock)
            log_message(f"Token log written: {out_jsonl}", log_file, log_lock)
            out_summary = Path(cfg.trace_out_dir) / "report" / "tokens_summary.json"
            log_message(f"Token summary written: {out_summary}", log_file, log_lock)
        except Exception as e:
            log_message(f"WARNING: failed to write trace report: {e}", log_file, log_lock)

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
