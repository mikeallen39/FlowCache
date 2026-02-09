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
TeaCache implementation for full output reuse.

This module provides TeaCache, which reuses all model outputs together when
the accumulated relative L1 distance is below threshold.
"""

import argparse
import gc
import sys
import torch
from types import MethodType

from inference.pipeline import MagiPipeline
from inference.pipeline.video_generate import SampleTransport, find_dit_model
from inference.pipeline.cache import TeaCache
from inference.pipeline.cache.utils import get_embedding_and_meta_with_chunk_info


def setup_teacache(
    rel_l1_thresh: float = 0.01,
    warmup_steps: int = 0,
    log: bool = False
):
    """
    Set up TeaCache for SampleTransport.

    Args:
        rel_l1_thresh: Relative L1 distance threshold for reuse
        warmup_steps: Number of warmup steps before reuse can happen
        log: Whether to log reuse decisions
    """
    # Create cache instance and attach to SampleTransport
    SampleTransport.cache_reuse_manager = TeaCache(
        rel_l1_thresh=rel_l1_thresh,
        warmup_steps=warmup_steps,
        log=log
    )

    # Monkey patch the SampleTransport methods
    SampleTransport.forward_velocity = teacache_forward_velocity
    SampleTransport.integrate_velocity = teacache_integrate_velocity


def teacache_forward_velocity(self, infer_idx: int, cur_denoise_step: int) -> torch.Tensor:
    """
    Forward pass with TeaCache output reuse.

    Args:
        self: SampleTransport instance
        infer_idx: Inference index
        cur_denoise_step: Current denoising step

    Returns:
        Velocity tensor
    """
    # Get cache from class attribute
    teacache = SampleTransport.cache_reuse_manager

    # 1. Get current work status
    x = self.xs[infer_idx]
    transport_input = self.transport_inputs[infer_idx]

    # 2. Extract denoising status
    (denoise_step_per_stage, denoise_stage, denoise_idx), (
        chunk_offset,
        chunk_start,
        chunk_end,
        t_start,
        t_end,
    ) = self.generate_denoise_status_and_sequences(infer_idx, cur_denoise_step)

    # 3. Prepare model kwargs
    model_kwargs = dict(
        chunk_width=self.chunk_width,
        fwd_extra_1st_chunk=False,
        num_steps=transport_input.num_steps
    )
    model_kwargs.update({
        "denoise_step_per_stage": denoise_step_per_stage,
        "denoise_stage": denoise_stage,
        "denoise_idx": denoise_idx
    })

    batch_size, chunk_token_nums = self.get_batch_size_and_chunk_token_nums(infer_idx)
    model_kwargs["chunk_token_nums"] = chunk_token_nums
    model_kwargs["chunk_num"] = transport_input.chunk_num
    model_kwargs["chunk_offset"] = chunk_offset

    if chunk_offset > 0 and cur_denoise_step == 0:
        self.extract_prefix_video_feature(
            infer_idx, transport_input.prefix_video, transport_input.y, chunk_offset, model_kwargs
        )

    # 4. Prepare inputs
    x_chunk = x[:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width].clone()
    y_chunk = transport_input.y[:, chunk_start:chunk_end]
    mask_chunk = transport_input.emb_masks[:, chunk_start:chunk_end]
    model_kwargs.update({
        "slice_point": chunk_start,
        "range_num": chunk_end,
        "denoising_range_num": chunk_end - chunk_start
    })

    # 5. Prepare timesteps
    denoise_step_of_each_chunk = self.get_denoise_step_of_each_chunk(
        infer_idx, denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=False
    )
    t = self.get_timestep(
        self.ts[infer_idx], denoise_step_per_stage, t_start, t_end, denoise_idx, has_clean_t=False
    )
    t = t.unsqueeze(0).repeat(x_chunk.size(0), 1)

    # 6. Generate KV range
    kv_range = self.generate_kvrange_for_denoising_video(
        infer_idx=infer_idx,
        slice_point=model_kwargs["slice_point"],
        denoising_range_num=model_kwargs["denoising_range_num"],
        denoise_step_of_each_chunk=denoise_step_of_each_chunk,
    )

    # 7. Pad prefix video if needed
    if transport_input.prefix_video is not None:
        x_chunk, t = self.try_pad_prefix_video(
            infer_idx, x_chunk, t, prefix_video_start=model_kwargs["slice_point"] * self.chunk_width
        )

    # 8. Model forward
    forward_fn = find_dit_model(self.model).forward_dispatcher
    nearly_clean_chunk_t = t[0, int(model_kwargs["fwd_extra_1st_chunk"])].item()
    model_kwargs["distill_nearly_clean_chunk"] = (
        nearly_clean_chunk_t > self.engine_config.distill_nearly_clean_chunk_threshold
    )
    model_kwargs["distill_interval"] = self.time_interval[infer_idx][denoise_idx]
    model_kwargs["total_num_steps"] = self.total_forward_step(infer_idx)

    # Initialize TeaCache step counter
    if teacache.cnt == 0 and teacache.num_steps == 0:
        teacache.num_steps = model_kwargs["total_num_steps"]

    # Setup monkey-patched model forward
    model = find_dit_model(self.model)
    model.forward = MethodType(_create_model_forward_fn(teacache), model)
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


def _create_model_forward_fn(teacache: TeaCache):
    """
    Create a model forward function with TeaCache logic.

    Args:
        teacache: TeaCache instance

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
    ) -> torch.Tensor:
        raw_x = x.clone()

        # 1. Compute feature metric
        metric_x = teacache.compute_feature_metric(
            x=x,
            x_embedder=model_self.x_embedder,
            x_rescale_factor=model_self.model_config.x_rescale_factor,
            half_channel_vae=model_self.model_config.half_channel_vae,
            params_dtype=model_self.model_config.params_dtype
        )

        # 2. Update kwargs with TeaCache state
        teacache.total_num_steps = kwargs['total_num_steps']
        denoise_step_per_stage = kwargs['denoise_step_per_stage']
        kwargs["start_chunk_id"] = kwargs['slice_point']
        kwargs["end_chunk_id"] = kwargs['range_num']
        kwargs['cur_denoise_step'] = teacache.cnt
        model_self.cur_denoise_step = teacache.cnt

        if kwargs.get("distill_nearly_clean_chunk", False):
            kwargs["end_chunk_id"] += 1

        # Handle nearly clean chunk (not used in TeaCache)
        if kwargs.get("fwd_extra_1st_chunk", False):
            metric_x = metric_x[kwargs["chunk_token_nums"]:, :, :]
        if kwargs.get("distill_nearly_clean_chunk", False):
            metric_x = metric_x[:-kwargs["chunk_token_nums"], :, :]

        # 3. Check if should reuse or calculate
        current_num_chunks = metric_x.shape[0] // kwargs["chunk_token_nums"]
        previous_num_chunks = (
            teacache.previous_modulated_input.shape[0] // kwargs["chunk_token_nums"]
            if teacache.previous_modulated_input is not None else 0
        )

        should_reuse = teacache.should_reuse(
            chunk_id=0,  # Not used in TeaCache
            step=teacache.cnt,
            current_features=metric_x,
            denoise_step_per_stage=denoise_step_per_stage,
            num_chunks_current=current_num_chunks,
            num_chunks_previous=previous_num_chunks
        )

        # 4. Handle partial reuse at stage boundary
        if (not should_reuse and
            teacache.cnt % denoise_step_per_stage == 0 and
            current_num_chunks > previous_num_chunks and
            teacache.accumulated_rel_l1_distance < teacache.rel_l1_thresh):

            # Only calculate new chunk
            range_num = kwargs['range_num'] - kwargs['chunk_offset']
            if kwargs.get("distill_nearly_clean_chunk", False):
                x = x[:, :, (range_num - 2) * kwargs['chunk_width']:(range_num - 1) * kwargs['chunk_width']]
                y = y[range_num - 2:range_num - 1]
                t = t[:, range_num - 2:range_num - 1]
                xattn_mask = xattn_mask[range_num - 2:range_num - 1]
                kwargs["start_chunk_id"] = kwargs['range_num'] - 2
                kwargs["end_chunk_id"] = kwargs['range_num'] - 1
                kwargs["denoising_range_num"] = 1
                model_self.discard_nearly_clean_chunk = True
            else:
                x = x[:, :, (range_num - 1) * kwargs['chunk_width']:range_num * kwargs['chunk_width']]
                y = y[range_num - 1:range_num]
                t = t[:, range_num - 1:range_num]
                xattn_mask = xattn_mask[range_num - 1:range_num]
                kwargs["start_chunk_id"] = kwargs['range_num'] - 1
                kwargs["denoising_range_num"] = 1

            model_self.single_chunk_inference = True
            model_self.denoising_range_num = kwargs["denoising_range_num"]

        # Store features for next step
        teacache.store_previous_features(metric_x)

        # 5. Forward or reuse
        if teacache.should_calc:
            (x, condition, condition_map, y_xattn_flat, rope, meta_args) = model_self.forward_pre_process(
                x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs
            )

            if not model_self.pre_process:
                from inference.pipeline.parallelism import pp_scheduler
                x = pp_scheduler().recv_prev_data(x.shape, x.dtype)
                model_self.videodit_blocks.set_input_tensor(x)
            else:
                x = x.clone()

            x = model_self.videodit_blocks.forward(
                hidden_states=x,
                condition=condition,
                condition_map=condition_map,
                y_xattn_flat=y_xattn_flat,
                rotary_pos_emb=rope,
                inference_params=inference_params,
                meta_args=meta_args,
            )

            if not model_self.post_process:
                from inference.pipeline.parallelism import pp_scheduler
                pp_scheduler().isend_next(x)

            return model_self.forward_post_process(x, meta_args)
        else:
            # Reuse: return zeros (output not used)
            return torch.zeros_like(raw_x)

    return model_forward


@torch.no_grad()
def _new_get_embedding_and_meta(
    model_self,
    x,
    t,
    y,
    caption_dropout_mask,
    xattn_mask,
    kv_range,
    **kwargs
):
    """Monkey-patched version of get_embedding_and_meta with chunk info."""
    return get_embedding_and_meta_with_chunk_info(
        model_self, x, t, y, caption_dropout_mask, xattn_mask, kv_range, **kwargs
    )


def teacache_integrate_velocity(self, infer_idx: int, cur_denoise_step: int):
    """
    Integrate velocity with TeaCache residual handling.

    Args:
        self: SampleTransport instance
        infer_idx: Inference index
        cur_denoise_step: Current denoising step
    """
    # Get cache from class attribute
    teacache = SampleTransport.cache_reuse_manager

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

    # Integrate with residual handling
    ori_x_chunk = x_chunk.clone()

    if teacache.should_calc:
        if velocity.shape[2] < x_chunk.shape[2]:
            # Partial reuse: only last chunk was computed
            t_num = x_chunk.shape[2] // self.chunk_width
            x_chunk = x_chunk[:, :, -self.chunk_width:]
            x_chunk = self.integrate(
                x_chunk, velocity, self.ts[infer_idx], denoise_step_per_stage,
                t_start, t_end, denoise_idx, delta_t_index=t_num - 1
            )
            # Concatenate with reused chunks
            x_chunk = torch.cat([teacache.previous_output, x_chunk], dim=2)
        else:
            # Full calculation
            x_chunk = self.integrate(
                x_chunk, velocity, self.ts[infer_idx], denoise_step_per_stage,
                t_start, t_end, denoise_idx
            )

        # Store residual for next step
        teacache.update_residual(0, x_chunk - ori_x_chunk)

        # Store output for potential next stage reuse
        if (teacache.cnt + 1) % denoise_step_per_stage == 0:
            teacache.previous_output = x_chunk
    else:
        # Reuse: add residual to input
        x_chunk = x_chunk + teacache.previous_residual[:, :, -x_chunk.shape[2]:]

    # Increment step counter
    teacache.increment_step()

    # Update chunk denoise count
    for chunk_index in range(chunk_start, chunk_end):
        chunk_denoise_count[chunk_index] += 1

    self.xs[infer_idx][:, :, chunk_start * self.chunk_width : chunk_end * self.chunk_width] = x_chunk
    self.chunk_denoise_count[infer_idx] = chunk_denoise_count

    # Return clean chunk if ready
    if chunk_denoise_count[chunk_start] == transport_input.num_steps:
        return _return_clean_chunk(
            self, infer_idx, transport_input, chunk_start, chunk_end, chunk_offset
        )

    return None, None


def _return_clean_chunk(self, infer_idx, transport_input, chunk_start, chunk_end, chunk_offset):
    """
    Return the clean chunk if denoising is complete.

    Args:
        self: SampleTransport instance
        infer_idx: Inference index
        transport_input: Transport input
        chunk_start: Start chunk ID
        chunk_end: End chunk ID
        chunk_offset: Prefix video offset

    Returns:
        Tuple of (clean_chunk, relative_chunk_id) or (None, None)
    """
    if transport_input.prefix_video is not None:
        prefix_video_length = transport_input.prefix_video.size(2)
        if (chunk_start + 1) * self.chunk_width <= prefix_video_length:
            return None, None

        real_start = max(chunk_start * self.chunk_width, prefix_video_length)

        # Keep the first 4-frames only for I2V Job
        if chunk_start == 0 and prefix_video_length == 1:
            real_start = 0

        clean_chunk, _ = self.xs[infer_idx][:, :, real_start:(chunk_start + 1) * self.chunk_width].chunk(2, dim=0)
        return clean_chunk, chunk_start - chunk_offset
    else:
        clean_chunk, _ = self.xs[infer_idx][
            :, :, chunk_start * self.chunk_width:(chunk_start + 1) * self.chunk_width
        ].chunk(2, dim=0)
        return clean_chunk, chunk_start - chunk_offset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MagiPipeline with TeaCache.")
    parser.add_argument('--config_file', type=str, help='Path to the configuration file.')
    parser.add_argument(
        '--mode', type=str, choices=['t2v', 'i2v', 'v2v'],
        required=True, help='Mode to run: t2v, i2v, or v2v.'
    )
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the pipeline.')
    parser.add_argument('--image_path', type=str, help='Path to the image file (for i2v mode).')
    parser.add_argument('--prefix_video_path', type=str, help='Path to the prefix video file (for v2v mode).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output video.')
    parser.add_argument('--use_teacache', action='store_true', help='Whether to use TeaCache.')
    parser.add_argument('--rel_l1_thresh', type=float, default=0.01, help='Relative L1 distance threshold.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps before reuse.')
    parser.add_argument('--log', action='store_true', help='Whether to log TeaCache information.')
    parser.add_argument('--print_peak_memory', action='store_true', help='Print peak memory usage.')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.print_peak_memory:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            device = torch.cuda.current_device()
            print(f"Running on GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory before pipeline: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        else:
            print("CUDA not available, running on CPU")

    print(f"TeaCache config: rel_l1_thresh={args.rel_l1_thresh}, "
          f"warmup_steps={args.warmup_steps}")

    # Setup TeaCache
    setup_teacache(
        rel_l1_thresh=args.rel_l1_thresh,
        warmup_steps=args.warmup_steps,
        log=args.log
    )

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
