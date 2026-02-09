# Copyright 2024 MAGI Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import argparse
import csv
import subprocess
from pathlib import Path

import multiprocessing as mp

# Constants
DEFAULT_BASE_PORT = 29510
PHYSICSIQ_FPS = 24


def load_yaml_config(yaml_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def apply_slice(items: list, start: int | None, end: int | None) -> list:
    """Apply start/end slice to a list with bounds checking."""
    if start is None and end is None:
        return items

    slice_start = max(0, start if start is not None else 0)
    slice_end = min(end if end is not None else len(items), len(items))
    slice_end = max(slice_start, slice_end)

    return items[slice_start:slice_end]


def configure_teacache(transport, config: dict) -> None:
    """Configure TeaCache reuse strategy on SampleTransport."""
    from inference.pipeline.teacache import (
        teacache_forward_velocity,
        teacache_integrate_velocity,
    )

    transport.rel_l1_thresh = config["rel_l1_thresh"]
    transport.accumulated_rel_l1_distance = 0
    transport.previous_modulated_input = None
    transport.previous_residual = None
    transport.cnt = 0
    transport.forward_velocity = teacache_forward_velocity
    transport.integrate_velocity = teacache_integrate_velocity
    transport.reuse_times = 0
    transport.warmup_steps = config["warmup_steps"]
    transport.previous_output = None
    transport.log = config.get("log", False)


def configure_kv_cache(transport, config: dict) -> None:
    """Configure KV cache compression if enabled."""
    if not config.get("compress_kv_cache", False):
        transport.compress_kv_cache = False
        return

    print("KV cache compression is enabled.")
    transport.compress_kv_cache = True

    assert config.get("total_cache_chunk_nums") is not None

    compression_config = {
        "method_config": {
            "compress_strategy": config["compress_strategy"],
            "mix_lambda": config["mix_lambda"],
            "query_granularity": config["query_granularity"],
            "score_weighting_method": config.get("score_weighting_method"),
            "power": config.get("power", 3),
        },
    }

    from inference.pipeline.kvcompress import replace_magi

    replace_magi(compression_config)


def configure_flowcache(transport, config: dict) -> None:
    """Configure FlowCache reuse strategy on SampleTransport."""
    from inference.pipeline.flowcache import (
        flowcache_forward_velocity,
        flowcache_integrate_velocity,
    )

    configure_kv_cache(transport, config)

    transport.rel_l1_thresh = config["rel_l1_thresh"]
    transport.chunk_accumulated_rel_l1 = 0
    transport.previous_modulated_input = None
    transport.previous_residual = None
    transport.cnt = 0
    transport.forward_velocity = flowcache_forward_velocity
    transport.integrate_velocity = flowcache_integrate_velocity
    transport.reuse_times = 0
    transport.warmup_steps = config["warmup_steps"]
    transport.previous_output = None
    transport.discard_nearly_clean_chunk = config.get("discard_nearly_clean_chunk", False)
    transport.chunk_accumulated_rel_l1 = None
    transport.prev_chunk_features = None
    transport.chunk_reuse_flags = None
    transport.total_cache_chunk_nums = config.get("total_cache_chunk_nums")
    transport.log = config.get("log", False)


def configure_reuse_strategy(config: dict) -> None:
    """Configure the appropriate reuse strategy on SampleTransport."""
    from inference.pipeline.video_generate import SampleTransport

    strategy = config["reuse_strategy"]

    if strategy == "original":
        return
    if strategy == "all":
        configure_teacache(SampleTransport, config)
    elif strategy == "chunkwise":
        configure_flowcache(SampleTransport, config)
    else:
        raise ValueError(f"Unknown reuse strategy: {strategy}")


def setup_environment(gpu_id: int) -> None:
    """Set up environment variables for a GPU worker process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(DEFAULT_BASE_PORT + gpu_id)

    # Enable pdb terminal debugging
    sys.stdin = open(0)


def filter_existing_samples(samples: list, config: dict) -> list:
    """Filter out samples whose output files already exist."""
    if config["benchmark"] == "vbench":
        return [
            sample
            for sample in samples
            if not os.path.exists(os.path.abspath(os.path.join(config["save_path"], f"{sample}-0.mp4")))
        ]
    else:  # physicsiq
        return [
            sample for sample in samples if not os.path.exists(sample["output_path"])
        ]


def assign_samples_to_gpu(
    samples: list, gpu_id: int, rank: int, num_gpus: int
) -> list:
    """Divide samples across GPUs and return the subset for this GPU."""
    samples_per_gpu = (len(samples) + num_gpus - 1) // num_gpus
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, len(samples))
    return samples[start_idx:end_idx]


def process_vbench_sample(pipeline, prompt: str, config: dict, gpu_id: int) -> None:
    """Process a single vbench text-to-video sample."""
    output_path = os.path.abspath(os.path.join(config["save_path"], f"{prompt}-0.mp4"))

    if os.path.exists(output_path):
        print(f"[SKIP GPU {gpu_id}] Already exists: {output_path}")
        return

    print(f"[GPU {gpu_id}] Generating T2V: '{prompt}' -> {output_path}")
    pipeline.run_text_to_video(prompt=prompt, output_path=output_path)
    print(f"[DONE GPU {gpu_id}] Saved: {output_path}")


def process_physicsiq_sample(pipeline, sample: dict, gpu_id: int) -> None:
    """Process a single PhysicsIQ video-to-video sample."""
    prompt = sample["description"]
    prefix_video_path = sample["prefix_video_path"]
    output_path = sample["output_path"]

    if not os.path.exists(prefix_video_path):
        print(f"[WARN GPU {gpu_id}] Conditioning video not found: {prefix_video_path}")
        return

    if os.path.exists(output_path):
        print(f"[SKIP GPU {gpu_id}] Already exists: {output_path}")
        return

    print(f"[GPU {gpu_id}] Generating V2V: '{prompt}'")
    print(f"   Input:  {prefix_video_path}")
    print(f"   Output: {output_path}")

    pipeline.run_video_to_video(
        prompt=prompt,
        prefix_video_path=prefix_video_path,
        output_path=output_path,
    )
    print(f"[DONE GPU {gpu_id}] Saved: {output_path}")


def worker_process(gpu_id: int, rank: int, config: dict, all_samples: list) -> None:
    """Independent worker running on each GPU."""
    setup_environment(gpu_id)
    configure_reuse_strategy(config)

    try:
        magi_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"]
        ).decode().strip()
        os.environ["MAGI_ROOT"] = magi_root
        os.environ["PYTHONPATH"] = f"{magi_root}:{os.environ.get('PYTHONPATH', '')}"
    except Exception as e:
        print(f"[GPU {gpu_id}] Failed to set MAGI_ROOT: {e}")
        return

    filtered_samples = filter_existing_samples(all_samples, config)

    if not filtered_samples:
        print(f"[GPU {gpu_id}] No samples need to be generated.")
        return

    print(f"Processing {len(filtered_samples)} samples.")

    my_samples = assign_samples_to_gpu(
        filtered_samples, gpu_id, rank, config["num_gpus"]
    )

    if not my_samples:
        print(f"[GPU {gpu_id}] No samples assigned.")
        return

    print(f"[GPU {gpu_id}] Assigned {len(my_samples)} samples")

    from inference.pipeline.entry import MagiPipeline

    print(f"[GPU {gpu_id}] Loading model...")
    pipeline = MagiPipeline(config["config_file"])
    print(f"[GPU {gpu_id}] Model loaded.")

    process_func = (
        process_vbench_sample if config["benchmark"] == "vbench" else process_physicsiq_sample
    )

    for sample in my_samples:
        process_func(pipeline, sample, config, gpu_id)

    print(f"[GPU {gpu_id}] Completed.")


def build_conditioning_video_path(
    data_root: str, vid_id: str, scenario: str, fps: int
) -> str:
    """Construct the path to the conditioning video file."""
    conditioning_dir = os.path.join(
        data_root, "physics-IQ-benchmark", "split-videos", "conditioning", f"{fps}FPS"
    )
    match_suffix = re.search(r"_(.*)", scenario)
    suffix = match_suffix.group(1) if match_suffix else ""
    filename = f"{vid_id}_conditioning-videos_{fps}FPS_{suffix}"
    return os.path.join(conditioning_dir, filename)


def load_physicsiq_samples(config: dict) -> list[dict]:
    """Load sample list from PhysicsIQ dataset."""
    data_root = config["physicsiq_data_dir"]
    descriptions_csv = os.path.join(data_root, "descriptions", "descriptions.csv")
    output_dir = config["save_path"]

    if not os.path.exists(descriptions_csv):
        raise FileNotFoundError(f"descriptions.csv not found at {descriptions_csv}")

    os.makedirs(output_dir, exist_ok=True)

    samples = []
    with open(descriptions_csv, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"].strip()
            match_id = re.match(r"^(\d+)_", scenario)

            if not match_id:
                print(f"Cannot extract ID from scenario: {scenario}")
                continue

            vid_id = match_id.group(1).zfill(4)
            description = row["description"]
            generated_video_name = row["generated_video_name"]
            prefix_video_path = build_conditioning_video_path(
                data_root, vid_id, scenario, PHYSICSIQ_FPS
            )
            output_path = os.path.join(output_dir, generated_video_name)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            samples.append({
                "vid_id": vid_id,
                "scenario": scenario,
                "description": description,
                "generated_video_name": generated_video_name,
                "prefix_video_path": prefix_video_path,
                "output_path": output_path,
            })

    # PhysicsIQ samples are duplicated; take only the first half
    unique_count = len(samples) // 2
    samples = samples[:unique_count]

    print(f"Loaded {unique_count} PhysicsIQ samples.")

    return apply_slice(samples, config.get("start"), config.get("end"))


def load_vbench_samples(config: dict) -> list[str]:
    """Load prompt list from vbench dimension file."""
    prompt_dir = config["vbench_prompt_dir"]
    dimension = config.get("dimension")

    if not dimension:
        raise ValueError("For vbench, 'dimension' must be specified in config")

    prompt_file = os.path.join(prompt_dir, f"{dimension}.txt")

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    return apply_slice(prompts, config.get("start"), config.get("end"))


def setup_save_path(config: dict) -> None:
    """Configure the output save path based on benchmark type."""
    base_path = config["base_save_path"]

    if config["benchmark"] == "vbench":
        dimension = config.get("dimension")
        videos_dir = os.path.join(base_path, "videos", dimension) if dimension else None
        config["save_path"] = videos_dir if videos_dir else os.path.join(base_path, "videos")
    elif config["benchmark"] == "physicsiq":
        config["save_path"] = os.path.join(base_path, "videos")

    os.makedirs(config["save_path"], exist_ok=True)


def main() -> None:
    """Entry point for video sampling script."""
    parser = argparse.ArgumentParser(
        description="Video sampling script using YAML configuration"
    )
    parser.add_argument("yaml_config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    config = load_yaml_config(args.yaml_config)
    print(f"Loaded configuration from: {args.yaml_config}")

    setup_save_path(config)

    gpu_ids = list(map(int, config["gpus"].split(",")))
    config["num_gpus"] = len(gpu_ids)

    benchmark = config["benchmark"]
    if benchmark == "vbench":
        all_samples = load_vbench_samples(config)
    elif benchmark == "physicsiq":
        data_root = config["physicsiq_data_dir"]
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        all_samples = load_physicsiq_samples(config)
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")

    print(f"Total samples: {len(all_samples)}")
    print(f"GPUs: {gpu_ids}")
    print(f"Output: {config['save_path']}")
    print(f"Config: {config['config_file']}")

    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(target=worker_process, args=(gpu_id, rank, config, all_samples))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
