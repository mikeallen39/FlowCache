# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""
FlowCache implementation: Per-chunk output reuse + KV cache compression.

This module provides FlowCache, which combines:
- ChunkWiseCache: Per-chunk output reuse for fine-grained control
- KVCacheCompressor: Dynamic KV cache compression for memory efficiency
"""

import argparse
import gc
import os
import sys
import torch
from types import MethodType

from inference.pipeline import MagiPipeline
from inference.pipeline.video_generate import SampleTransport, find_dit_model
from inference.pipeline.cache import ChunkWiseCache, KVCacheCompressor
from inference.pipeline.cache.utils import (
    generate_dynamic_kv_range,
    get_embedding_and_meta_with_chunk_info,
)
from inference.pipeline.kvcompress import replace_magi
from inference.pipeline.kvcompress.utils import ChunkKVRangeTracker


def setup_flowcache(
    rel_l1_thresh: float = 0.01,
    warmup_steps: int = 0,
    discard_nearly_clean_chunk: bool = False,
    log: bool = False,
    total_cache_chunk_nums: int = 5,
    compress_kv_cache: bool = True,
):
    """
    Set up FlowCache with per-chunk reuse and KV compression.

    Args:
        rel_l1_thresh: Relative L1 distance threshold for reuse
        warmup_steps: Number of warmup steps per chunk before reuse can happen
        discard_nearly_clean_chunk: Whether to skip nearly-clean chunk
        log: Whether to log reuse decisions
        total_cache_chunk_nums: Total number of chunks to cache
        compress_kv_cache: Whether to enable KV cache compression
    """
    # Create cache instance and attach to SampleTransport
    SampleTransport.cache_reuse_manager = ChunkWiseCache(
        rel_l1_thresh=rel_l1_thresh,
        warmup_steps=warmup_steps,
        discard_nearly_clean_chunk=discard_nearly_clean_chunk,
        log=log
    )

    # Initialize compressor placeholder (will be created at runtime)
    SampleTransport.kv_compress_manager = None

    # Monkey patch the SampleTransport methods
    SampleTransport.forward_velocity = flowcache_forward_velocity
    SampleTransport.integrate_velocity = flowcache_integrate_velocity
    SampleTransport.total_cache_chunk_nums = total_cache_chunk_nums
    SampleTransport.compress_kv_cache = compress_kv_cache


def flowcache_forward_velocity(self, infer_idx: int, cur_denoise_step: int) -> dict:
    """
    Forward pass with per-chunk TeaCache and KV compression.

    Args:
        self: SampleTransport instance
        infer_idx: Inference index
        cur_denoise_step: Current denoising step

    Returns:
        Dictionary mapping chunk_id to velocity tensor
    """
    # Get cache from class attribute
    cache = SampleTransport.cache_reuse_manager

    # 1. Get current work status
    x = self.xs[infer_idx]
    transport_input = self.transport_inputs[infer_idx]
    batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)

    # 2. Initialize KV cache tracking if needed
    if hasattr(self, 'compress_kv_cache') and self.compress_kv_cache:
        total_cache_len = self.total_cache_chunk_nums * (
            self.chunk_width *
            (transport_input.latent_size[3] // self.model_config.patch_size) *
            (transport_input.latent_size[4] // self.model_config.patch_size)
        )

        if not hasattr(self.inference_params[infer_idx], 'kv_chunk_tracker'):
            self.inference_params[infer_idx].kv_chunk_tracker = ChunkKVRangeTracker(
                total_cache_len=total_cache_len,
                clip_token_nums=chunk_token_nums,
                max_batch_size=1
            )

        if not hasattr(self, 'chunk_query_states'):
            self.chunk_query_states = {}

    # 3. Initialize chunk state
    cache.initialize_chunk_state(transport_input.chunk_num)

    # 4. Extract denoising status
    (denoise_step_per_stage, denoise_stage, denoise_idx), (
        chunk_offset,
        chunk_start,
        chunk_end,
        t_start,
        t_end,
    ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)

    # 5. Prepare model kwargs
    model_kwargs = dict(
        chunk_width=self.chunk_width,
        fwd_extra_1st_chunk=False,
        num_steps=transport_input.num_steps
    )
    if hasattr(self, "debug"):
        model_kwargs["debug"] = self.debug
    model_kwargs.update({
        "denoise_step_per_stage": denoise_step_per_stage,
        "denoise_stage": denoise_stage,
        "denoise_idx": denoise_idx,
        "chunk_num": transport_input.chunk_num
    })

    if hasattr(self, 'compress_kv_cache') and self.compress_kv_cache:
        model_kwargs.update({
            "compress_kv": True,
            "total_cache_len": total_cache_len
        })
    else:
        model_kwargs["save_kvcache_every_forward"] = True
        
    if chunk_offset > 0 and cur_denoise_step == 0:
        self.extract_prefix_video_feature(
            infer_idx, transport_input.prefix_video, transport_input.y, chunk_offset, model_kwargs
        )

    # 6. Prepare inputs
    x_chunk = x[:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width].clone()
    y_chunk = transport_input.y[:, chunk_start:chunk_end]
    mask_chunk = transport_input.emb_masks[:, chunk_start:chunk_end]
    model_kwargs.update({
        "slice_point": chunk_start,
        "range_num": chunk_end,
        "denoising_range_num": chunk_end - chunk_start
    })
    model_kwargs["chunk_token_nums"] = chunk_token_nums

    # 7. Prepare timesteps
    denoise_step_of_each_chunk = self.get_denoise_step_of_each_chunk(
        infer_idx, denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=False
    )
    t = self.get_timestep(
        self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=False
    )
    t = t.unsqueeze(0).repeat(x_chunk.size(0), 1)

    # 8. Generate KV range
    kv_range = self.generate_kvrange_for_denoising_video(
        infer_idx=infer_idx,
        slice_point=model_kwargs["slice_point"],
        denoising_range_num=model_kwargs["denoising_range_num"],
        denoise_step_of_each_chunk=denoise_step_of_each_chunk,
    )

    # 9. Pad prefix video if needed
    if transport_input.prefix_video is not None:
        x_chunk, t = self.try_pad_prefix_video(
            infer_idx, x_chunk, t, prefix_video_start=model_kwargs["slice_point"] * self.chunk_width
        )

    # 10. Model forward
    forward_fn = find_dit_model(self.model).forward_dispatcher
    nearly_clean_chunk_t = t[0, int(model_kwargs["fwd_extra_1st_chunk"])].item()
    model_kwargs["distill_nearly_clean_chunk"] = (
        nearly_clean_chunk_t > self.engine_config.distill_nearly_clean_chunk_threshold
    )
    model_kwargs["distill_interval"] = self.time_interval[infer_idx][denoise_idx]
    model_kwargs["total_num_steps"] = self.total_forward_step(infer_idx)

    # Initialize step counter
    cache.set_total_steps(model_kwargs["total_num_steps"])

    # Setup monkey-patched model forward
    model = find_dit_model(self.model)
    model.forward = MethodType(_create_flowcache_model_forward_fn(cache, self, infer_idx), model)
    model.get_embedding_and_meta = MethodType(_new_get_embedding_and_meta, model)

    velocity = forward_fn(
        x=x_chunk,
        timestep=t,
        y=y_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1),
        mask=mask_chunk.flatten(start_dim=0, end_dim=1).unsqueeze(1),
        kv_range=kv_range,
        inference_params=self.inference_params[infer_idx],
        **model_kwargs,
    )

    self.x_chunks[infer_idx] = x_chunk
    self.velocities[infer_idx] = velocity
    return velocity


def _create_flowcache_model_forward_fn(cache: ChunkWiseCache, transport, infer_idx: int):
    """
    Create a model forward function with per-chunk cache and KV compression logic.

    Args:
        cache: ChunkWiseCache instance
        transport: SampleTransport instance
        infer_idx: Inference index

    Returns:
        Model forward function
    """
    @torch.no_grad()
    def model_forward(
        model_self,
        x,
        t,
        y,
        caption_dropout_mask=None,
        xattn_mask=None,
        kv_range=None,
        inference_params=None,
        **kwargs,
    ) -> dict:
        raw_x = x.clone()

        # 1. Compute feature metrics per chunk
        # Following source code: compute metric_x first, handle slicing, then split
        metric_chunks, num_chunks = cache.compute_feature_metric(
            x=x,
            x_embedder=model_self.x_embedder,
            x_rescale_factor=model_self.model_config.x_rescale_factor,
            half_channel_vae=model_self.model_config.half_channel_vae,
            chunk_token_nums=kwargs["chunk_token_nums"],
            params_dtype=model_self.model_config.params_dtype,
            offset=kwargs['slice_point'],
            fwd_extra_1st_chunk=kwargs.get("fwd_extra_1st_chunk", False),
            distill_nearly_clean_chunk=kwargs.get("distill_nearly_clean_chunk", False)
        )

        # 2. Update kwargs
        cache.total_num_steps = kwargs['total_num_steps']
        denoise_step_per_stage = kwargs['denoise_step_per_stage']
        kwargs['cur_denoise_step'] = cache.cnt
        model_self.cur_denoise_step = cache.cnt

        # 3. Split x into chunks (using num_chunks from metric_x, matching source code)
        chunk_width = kwargs["chunk_width"]
        offset = kwargs['slice_point']
        x_chunks = {}
        # Artifact chunks in x are not included - following source code comment
        for i in range(num_chunks):
            start_idx = i * chunk_width
            end_idx = start_idx + chunk_width
            x_chunks[offset + i] = x[:, :, start_idx:end_idx]

        # 4. Handle nearly clean chunk (artifact chunk) - add separately AFTER normal chunks
        # Following source code logic
        model_self.discard_nearly_clean_chunk = cache.discard_nearly_clean_chunk
        near_clean_chunk_idx = -1
        if not cache.discard_nearly_clean_chunk and kwargs.get("distill_nearly_clean_chunk", False):
            # Add artifact chunk - following source code comment
            near_clean_chunk_idx = max(x_chunks.keys()) + 1
            model_self.near_clean_chunk_idx = near_clean_chunk_idx
            x_chunks[near_clean_chunk_idx] = x[:, :, -chunk_width:]

        # 5. Determine which chunks to reuse
        if cache.cnt != 0 and cache.cnt != cache.num_steps - 1:
            current_num_chunks = len(metric_chunks)
            previous_num_chunks = len(cache.prev_metric_chunks)

            common_keys = set(metric_chunks.keys()) & set(cache.prev_metric_chunks.keys())
            for i in sorted(common_keys):
                should_reuse = cache.should_reuse(
                    chunk_id=i,
                    step=cache.cnt,
                    current_features=metric_chunks,
                    chunk_denoise_count=transport.chunk_denoise_count[infer_idx],
                    current_num_chunks=current_num_chunks,
                    previous_num_chunks=previous_num_chunks
                )
                cache.chunk_reuse_flags[i] = should_reuse

        # 6. Remove nearly clean chunk if first chunk can be reused
        if cache.chunk_reuse_flags.get(kwargs["slice_point"], False) and near_clean_chunk_idx != -1:
            x_chunks.pop(near_clean_chunk_idx, None)

        # 7. Store previous features
        cache.store_previous_features(metric_chunks)

        # 8. Forward chunks that are not reused
        current_infer_outputs = {}

        for i in sorted(x_chunks.keys()):
            if i in cache.chunk_reuse_flags and cache.chunk_reuse_flags[i]:
                continue

            x_i = x_chunks[i]
            # Handle near_clean_chunk_idx: use last chunk of t, y, xattn_mask
            if i == near_clean_chunk_idx:
                t_i = t[:, -1:]
                y_i = y[-1:]
                xattn_mask_i = xattn_mask[-1:]
            else:
                t_i = t[:, i - offset:i - offset + 1]
                y_i = y[i - offset:i - offset + 1]
                xattn_mask_i = xattn_mask[i - offset:i - offset + 1]

            kwargs["start_chunk_id"] = i
            kwargs["end_chunk_id"] = i + 1
            kwargs["denoising_range_num"] = 1

            if i == near_clean_chunk_idx:
                kwargs["distill_nearly_clean_chunk"] = True
            else:
                kwargs["distill_nearly_clean_chunk"] = False

            # Update KV range if compressed
            if hasattr(transport, 'compress_kv_cache') and transport.compress_kv_cache:
                if inference_params.kv_compressed:
                    kv_range = generate_dynamic_kv_range(
                        tracker=inference_params.kv_chunk_tracker,
                        current_chunk_id=i,
                        x_chunks_keys=list(x_chunks.keys()),
                        chunk_token_nums=kwargs["chunk_token_nums"],
                        near_clean_chunk_idx=near_clean_chunk_idx
                    )

            kwargs["near_clean_chunk_idx"] = near_clean_chunk_idx
            (processed_x, condition, condition_map, y_xattn_flat, rope, meta_args) = \
                model_self.forward_pre_process(
                    x_i, t_i, y_i, caption_dropout_mask, xattn_mask_i, kv_range, **kwargs
                )

            if not model_self.pre_process:
                from inference.pipeline.parallelism import pp_scheduler
                processed_x = pp_scheduler().recv_prev_data(processed_x.shape, processed_x.dtype)
                model_self.videodit_blocks.set_input_tensor(processed_x)
            else:
                processed_x = processed_x.clone()

            try:
                out = model_self.videodit_blocks.forward(
                    hidden_states=processed_x,
                    condition=condition,
                    condition_map=condition_map,
                    y_xattn_flat=y_xattn_flat,
                    rotary_pos_emb=rope,
                    inference_params=inference_params,
                    meta_args=meta_args,
                )
            except Exception as e:
                import pdb; pdb.set_trace()

            # Store query states for compression
            if hasattr(transport, 'compress_kv_cache') and transport.compress_kv_cache:
                for layer in model_self.videodit_blocks.layers:
                    layer_num = layer.self_attention.layer_number
                    if hasattr(layer.self_attention, '_last_query'):
                        transport.chunk_query_states[layer_num] = layer.self_attention._last_query

            if not model_self.post_process:
                from inference.pipeline.parallelism import pp_scheduler
                pp_scheduler().isend_next(out)

            out = model_self.forward_post_process(out, meta_args)
            current_infer_outputs[i] = out.clone().detach()

        return current_infer_outputs

    return model_forward


@torch.no_grad()
def _new_get_embedding_and_meta(model_self, x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs):
    """Monkey-patched version of get_embedding_and_meta with chunk info."""
    return get_embedding_and_meta_with_chunk_info(
        model_self, x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs
    )


def flowcache_integrate_velocity(self, infer_idx: int, cur_denoise_step: int):
    """
    Integrate velocity with per-chunk cache residual handling and KV compression.

    Args:
        self: SampleTransport instance
        infer_idx: Inference index
        cur_denoise_step: Current denoising step
    """
    # Get cache from class attribute
    cache = SampleTransport.cache_reuse_manager

    transport_input = self.transport_inputs[infer_idx]
    x_chunk = self.x_chunks[infer_idx]
    velocity = self.velocities[infer_idx]
    chunk_denoise_count = self.chunk_denoise_count[infer_idx]

    (denoise_step_per_stage, denoise_stage, denoise_idx), (
        chunk_offset,
        chunk_start,
        chunk_end,
        t_start,
        t_end,
    ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)

    chunk_num = x_chunk.shape[2] // self.chunk_width
    offset = chunk_start
    ori_x_chunk = x_chunk.clone()

    # Split into chunks
    x_chunks = {}
    for i in range(chunk_num):
        start_idx = i * self.chunk_width
        end_idx = start_idx + self.chunk_width
        x_chunks[offset + i] = x_chunk[:, :, start_idx:end_idx]

    # Integrate per chunk
    for i in range(chunk_num):
        if cache.chunk_reuse_flags[offset + i]:
            # Reuse: add residual
            x_chunk[:, :, i * self.chunk_width:(i + 1) * self.chunk_width] += \
                cache.previous_residual[offset + i]
        else:
            # Recalculate
            assert (offset + i) in velocity, f"Chunk {offset + i} not in velocity outputs"
            x_chunk[:, :, i * self.chunk_width:(i + 1) * self.chunk_width] = \
                self.integrate(x_chunks[offset + i], velocity[offset + i], self.ts[infer_idx],
                              denoise_step_per_stage, t_start, t_end, denoise_idx, i)
            # Store residual
            cache.previous_residual[offset + i] = \
                x_chunk[:, :, i * self.chunk_width:(i + 1) * self.chunk_width] - \
                ori_x_chunk[:, :, i * self.chunk_width:(i + 1) * self.chunk_width]

    # Increment step counter
    cache.increment_step()

    # Update chunk denoise count
    for chunk_index in range(chunk_start, chunk_end):
        chunk_denoise_count[chunk_index] += 1

    self.xs[infer_idx][:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width] = x_chunk
    self.chunk_denoise_count[infer_idx] = chunk_denoise_count

    # Check if KV compression is needed
    if hasattr(self, 'compress_kv_cache') and self.compress_kv_cache:
        _check_and_compress_kv(self, infer_idx, chunk_start, transport_input)

    # Return clean chunk if ready
    if chunk_denoise_count[chunk_start] == transport_input.num_steps:
        return _return_clean_chunk(self, infer_idx, transport_input, chunk_start, chunk_end, chunk_offset)

    return None, None


def _check_and_compress_kv(self, infer_idx: int, chunk_start: int, transport_input):
    """Check and perform KV cache compression if needed."""
    inference_params = self.inference_params[infer_idx]
    tracker = inference_params.kv_chunk_tracker

    total_cache_len = self.total_cache_chunk_nums * (
        self.chunk_width *
        (transport_input.latent_size[3] // self.model_config.patch_size) *
        (transport_input.latent_size[4] // self.model_config.patch_size)
    )

    # Get or create compressor from class attribute
    compressor = SampleTransport.kv_compress_manager
    if compressor is None:
        chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)[1]
        compressor = KVCacheCompressor(
            total_cache_len=total_cache_len,
            tokens_per_chunk=chunk_token_nums,
            budget_chunk_nums=self.total_cache_chunk_nums - 1,
            window_size=self.window_size
        )
        SampleTransport.kv_compress_manager = compressor

    # Check if compression needed
    if compressor.should_compress(
        tracker=tracker,
        chunk_num=transport_input.chunk_num,
        chunk_start=chunk_start,
        transport_input=transport_input,
        chunk_denoise_count=self.chunk_denoise_count[infer_idx]
    ):
        compressor.compress(
            model=find_dit_model(self.model),
            inference_params=inference_params,
            tracker=tracker,
            transport_input=transport_input,
            chunk_start=chunk_start,
            chunk_denoise_count=self.chunk_denoise_count[infer_idx],
            query_states_dict=self.chunk_query_states
        )


def _return_clean_chunk(self, infer_idx, transport_input, chunk_start, chunk_end, chunk_offset):
    """Return the clean chunk if denoising is complete."""
    if transport_input.prefix_video is not None:
        prefix_video_length = transport_input.prefix_video.size(2)
        if (chunk_start + 1) * self.chunk_width <= prefix_video_length:
            return None, None

        real_start = max(chunk_start * self.chunk_width, prefix_video_length)

        if chunk_start == 0 and prefix_video_length == 1:
            real_start = 0

        clean_chunk, _ = self.xs[infer_idx][:, :, real_start:(chunk_start + 1) * self.chunk_width].chunk(2, dim=0)
        return clean_chunk, chunk_start - chunk_offset
    else:
        clean_chunk, _ = self.xs[infer_idx][
            :, :, chunk_start * self.chunk_width:(chunk_start + 1) * self.chunk_width
        ].chunk(2, dim=0)
        return clean_chunk, chunk_start - chunk_offset


def load_config(config_path: str) -> dict:
    """Load configuration from JSON or YAML file."""
    _, ext = os.path.splitext(config_path)
    with open(config_path, 'r') as f:
        if ext == '.json':
            import json
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MagiPipeline with FlowCache.")
    parser.add_argument('--config_file', type=str, help='Path to the configuration file.')
    parser.add_argument(
        '--mode', type=str, choices=['t2v', 'i2v', 'v2v'],
        required=True, help='Mode to run: t2v, i2v, or v2v.'
    )
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the pipeline.')
    parser.add_argument('--image_path', type=str, help='Path to the image file (for i2v mode).')
    parser.add_argument('--prefix_video_path', type=str, help='Path to the prefix video file (for v2v mode).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output video.')
    parser.add_argument('--additional_config', type=str, help='Path to additional config file.')
    parser.add_argument('--print_peak_memory', action='store_true', help='Print peak memory usage.')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Load additional config
    if args.additional_config:
        additional_config = load_config(args.additional_config)
        print(f"Loading additional config: {additional_config}")

        for key, value in additional_config.items():
            setattr(args, key, value)
            print(f"Added to args: {key} = {value}")

        # Handle parameter name compatibility
        if hasattr(args, 'no_reuse_first_n_steps') and not hasattr(args, 'warmup_steps'):
            args.warmup_steps = args.no_reuse_first_n_steps
        if hasattr(args, 'no_reuse_mode'):
            # no_reuse_mode is deprecated, ignore it
            pass
    else:
        print("No additional config provided.")

    if args.print_peak_memory:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            device = torch.cuda.current_device()
            print(f"Running on GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory before pipeline: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        else:
            print("CUDA not available, running on CPU")

    # Setup FlowCache
    setup_flowcache(
        rel_l1_thresh=args.rel_l1_thresh,
        warmup_steps=args.warmup_steps,
        discard_nearly_clean_chunk=args.discard_nearly_clean_chunk,
        log=args.log,
        total_cache_chunk_nums=args.total_cache_chunk_nums,
        compress_kv_cache=args.compress_kv_cache,
    )

    # Setup KV compression in model
    compression_config = {
        "method_config": {
            "compress_strategy": getattr(args, 'compress_strategy', 'token'),
            "mix_lambda": getattr(args, 'mix_lambda', 0.07),
            "query_granularity": getattr(args, 'query_granularity', 'chunk'),
            "score_weighting_method": getattr(args, 'score_weighting_method', 'no_weight'),
            "power": getattr(args, 'power', 3),
        },
    }
    replace_magi(compression_config)

    # Run pipeline
    pipeline = MagiPipeline(args.config_file)

    if args.mode == 't2v':
        pipeline.run_text_to_video(prompt=args.prompt, output_path=args.output_path)
    elif args.mode == 'i2v':
        if not args.image_path:
            print("Error: --image_path is required for i2v mode.")
            sys.exit(1)
        pipeline.run_image_to_video(prompt=args.prompt, image_path=args.image_path, output_path=args.output_path)
    elif args.mode == 'v2v':
        if not args.prefix_video_path:
            print("Error: --prefix_video_path is required for v2v mode.")
            sys.exit(1)
        pipeline.run_video_to_video(
            prompt=args.prompt, prefix_video_path=args.prefix_video_path, output_path=args.output_path
        )

    if args.print_peak_memory:
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            cached_memory = torch.cuda.memory_reserved(device) / 1024**3
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3

            print("\n" + "=" * 50)
            print("GPU Memory Usage Summary:")
            print(f"Peak memory allocated: {peak_memory:.2f} GB")
            print(f"Current memory allocated: {current_memory:.2f} GB")
            print(f"Cached memory reserved: {cached_memory:.2f} GB")
            print(f"Total GPU memory: {total_memory:.2f} GB")
            print(f"Peak memory usage: {(peak_memory/total_memory)*100:.1f}%")
            print("=" * 50)

            gc.collect()
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(device) / 1024**3
            print(f"Memory after cache cleanup: {final_memory:.2f} GB")


if __name__ == "__main__":
    main()
