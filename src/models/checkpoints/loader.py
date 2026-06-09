import math
import torch


def load_model_state(module, state_dict, strict: bool, label: str):
    result = module.load_state_dict(state_dict, strict=strict)
    print(f"Loading {label} weights", result)
    if not strict:
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        if missing:
            print(f"{label} missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"{label} unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    return result


def interpolate_pos_embed_tensor(source_tensor: torch.Tensor, target_shape) -> torch.Tensor:
    if source_tensor.ndim != 3 or len(target_shape) != 3:
        raise ValueError("pos_embed interpolation expects 3D tensors")

    source_frames, source_patches, hidden = source_tensor.shape
    target_frames, target_patches, target_hidden = target_shape
    if source_frames != target_frames or hidden != target_hidden:
        raise ValueError(
            f"Cannot interpolate pos_embed from {tuple(source_tensor.shape)} to {tuple(target_shape)}"
        )

    source_hw = int(math.isqrt(source_patches))
    target_hw = int(math.isqrt(target_patches))
    if source_hw * source_hw != source_patches or target_hw * target_hw != target_patches:
        raise ValueError(
            f"pos_embed patch counts must be perfect squares, got {source_patches} -> {target_patches}"
        )

    pos_embed = source_tensor.reshape(source_frames, source_hw, source_hw, hidden).permute(0, 3, 1, 2)
    pos_embed = torch.nn.functional.interpolate(
        pos_embed,
        size=(target_hw, target_hw),
        mode="bicubic",
        align_corners=False,
    )
    return pos_embed.permute(0, 2, 3, 1).reshape(target_frames, target_patches, hidden)


def prepare_checkpoint_state_dict(
    module,
    state_dict,
    *,
    label: str,
    ignore_keys=(),
    ignore_shape_mismatch: bool = False,
    interpolate_pos_embed: bool = False,
):
    target_state = module.state_dict()
    prepared_state = {}
    skipped_keys = []
    skipped_mismatches = []
    interpolated_keys = []

    for raw_name, value in state_dict.items():
        name = raw_name.replace('_orig_mod.', '')
        if any(name == key or name.startswith(f"{key}.") for key in ignore_keys):
            skipped_keys.append(name)
            continue

        target_value = target_state.get(name)
        if target_value is None:
            prepared_state[name] = value
            continue

        if value.shape == target_value.shape:
            prepared_state[name] = value
            continue

        if interpolate_pos_embed and name == "pos_embed":
            prepared_state[name] = interpolate_pos_embed_tensor(value, target_value.shape)
            interpolated_keys.append((name, tuple(value.shape), tuple(target_value.shape)))
            continue

        if ignore_shape_mismatch:
            skipped_mismatches.append((name, tuple(value.shape), tuple(target_value.shape)))
            continue

        prepared_state[name] = value

    if skipped_keys:
        print(f"{label} skipped keys ({len(skipped_keys)}): {skipped_keys[:8]}")
    if skipped_mismatches:
        preview = [f"{name}: {src} -> {dst}" for name, src, dst in skipped_mismatches[:8]]
        print(f"{label} skipped shape mismatches ({len(skipped_mismatches)}): {preview}")
    if interpolated_keys:
        preview = [f"{name}: {src} -> {dst}" for name, src, dst in interpolated_keys[:8]]
        print(f"{label} interpolated keys ({len(interpolated_keys)}): {preview}")

    return prepared_state
