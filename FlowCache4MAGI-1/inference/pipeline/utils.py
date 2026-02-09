import torch

def get_tensors_memory_usage(tensor_list, unit='MB'):
    total_bytes = 0
    for t in tensor_list:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            total_bytes += t.element_size() * t.numel()

    unit = unit.upper()
    scale_dict = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
    }

    if unit not in scale_dict:
        raise ValueError(f"Unsupported unit: {unit}. Use 'B', 'KB', 'MB', or 'GB'.")

    scale = scale_dict[unit]
    return total_bytes / scale