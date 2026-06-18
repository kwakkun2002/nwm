import os

from src.core.paths import DEFAULT_WEIGHTS_ROOT

DEFAULT_LOCAL_VAE_PATH = os.path.join(DEFAULT_WEIGHTS_ROOT, "pretrained", "vae", "sd-vae-ft-ema")


def resolve_vae_source(local_path: str = DEFAULT_LOCAL_VAE_PATH, hf_name: str = "stabilityai/sd-vae-ft-ema") -> str:
    if os.path.isdir(local_path):
        return local_path
    return hf_name


def load_vae(device, local_path: str = DEFAULT_LOCAL_VAE_PATH, hf_name: str = "stabilityai/sd-vae-ft-ema"):
    from diffusers.models import AutoencoderKL

    vae_source = resolve_vae_source(local_path=local_path, hf_name=hf_name)
    return AutoencoderKL.from_pretrained(vae_source).to(device)
