import argparse
import torch
import json, os
import time

from scandl import sp_gaussian_diffusion as gd
from scandl.sp_gaussian_diffusion import SpacedDiffusion, space_timesteps
from scandl.sp_transformer_model_ablation import TransformerNetModel


def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('scandl/config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    num_transformer_layers,
    num_transformer_heads,
    mask_padding,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    one_noise_step,
    nll_in_loss,
    notes,
    **kwargs,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        num_transformer_layers=num_transformer_layers,
        num_transformer_heads=num_transformer_heads,
        one_noise_step=one_noise_step,
        mask_padding=mask_padding,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas=learn_sigma,
        sigma_small=sigma_small,
        use_kl=use_kl,
        one_noise_step=one_noise_step,
        nll_in_loss=nll_in_loss,
        mask_padding=mask_padding,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
