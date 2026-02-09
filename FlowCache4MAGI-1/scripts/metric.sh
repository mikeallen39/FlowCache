export CUDA_VISIBLE_DEVICES=3

python tools/video_metrics.py \
--original_video "/path/to/original_video.mp4" \
--generated_video "/path/to/generated_video.mp4"