from typing import Literal, Optional

import torch
from careamics.config import VAEBasedAlgorithm
from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.loss_model import KLLossConfig, LVAELossConfig
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.optimizer_models import LrSchedulerModel, OptimizerModel
from careamics.config.training_model import TrainingConfig


def get_model_config(**kwargs) -> LVAEModel:
    """Get the model configuration for the MicroSplit model.

    Parameters
    ----------
    ...

    Returns
    -------
    LVAEModel
        The LVAE model configuration.
    """
    return LVAEModel(
        architecture="LVAE",
        input_shape=kwargs["img_size"],
        encoder_conv_strides=kwargs["encoder_conv_strides"],
        decoder_conv_strides=kwargs["decoder_conv_strides"],
        multiscale_count=kwargs["multiscale_count"],
        z_dims=[128, 128, 128, 128],
        output_channels=kwargs["target_channels"],
        predict_logvar=kwargs["predict_logvar"],
        analytical_kl=False,
    )


def get_likelihood_config(
    **kwargs,
) -> tuple[GaussianLikelihoodConfig, MultiChannelNMConfig, NMLikelihoodConfig]:
    """Get the likelihood configuration for split models.

    Parameters
    ----------
    ...

    Returns
    -------
    tuple[GaussianLikelihoodConfig, MultiChannelNMConfig, NMLikelihoodConfig]
        The likelihoods and noise models configurations.
    """
    # gaussian likelihood
    if kwargs["loss_type"] in ["musplit", "denoisplit_musplit"]:
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=kwargs["predict_logvar"],
            logvar_lowerbound=-5.0,
        )
    else:
        gaussian_lik_config = None
    # noise model likelihood
    if kwargs["loss_type"] in ["denoisplit", "denoisplit_musplit"]:
        gmm_list = []
        for NM_path in kwargs["nm_paths"]:
            gmm_list.append(
                GaussianMixtureNMConfig(
                    model_type="GaussianMixtureNoiseModel",
                    path=NM_path,
                )
            )
        noise_model_config = MultiChannelNMConfig(noise_models=gmm_list)
        nm_lik_config = NMLikelihoodConfig(
            data_mean=kwargs["data_stats"][0],
            data_std=kwargs["data_stats"][1],
        )
    else:
        noise_model_config = None
        nm_lik_config = None

    return gaussian_lik_config, noise_model_config, nm_lik_config


def get_loss_config(**kwargs) -> LVAELossConfig:
    """Get the loss configuration for the split model.

    Parameters
    ----------
    ...

    Returns
    -------
    LVAELossConfig
        The loss configuration.
    """
    return LVAELossConfig(
        loss_type=kwargs["loss_type"],
        kl_params=KLLossConfig(
            loss_type=kwargs["kl_type"],
        ),
    )


def get_training_config(**kwargs) -> TrainingConfig:
    """Get the training configuration.

    Parameters
    ----------
    ...

    Returns
    -------
    TrainingConfig
        The training configuration.
    """
    # AMP on non-CUDA backends can emit CUDA autocast warnings (e.g. on MPS),
    # so default to mixed precision only when CUDA is available.
    if kwargs.get("precision") is not None:
        precision = kwargs["precision"]
    elif torch.cuda.is_available():
        precision = "16-mixed"
    else:
        precision = "32"

    return TrainingConfig(
        num_epochs=kwargs["num_epochs"],
        precision=precision,
        logger="wandb",
        gradient_clip_algorithm="value",
        grad_clip_norm_value=0.5,
        lr=kwargs["lr"],
        lr_scheduler_patience=kwargs["lr_scheduler_patience"],
    )


def get_lr_scheduler_config(**kwargs) -> LrSchedulerModel:
    return LrSchedulerModel(
        name="ReduceLROnPlateau",
        parameters={
            "mode": "min",
            "factor": 0.5,
            "patience": kwargs["lr_scheduler_patience"],
            "verbose": True,
            "min_lr": 1e-12,
        },
    )


def get_optimizer_config(**kwargs) -> OptimizerModel:
    return OptimizerModel(
        name="Adamax",
        parameters={
            "lr": kwargs["lr"],
            "weight_decay": 0,
        },
    )


# TODO: Some of these are typed as optional but VAEBasedAlgorithm does not agree...
def create_algorithm_config(
    algorithm: Literal["muspit", "denoisplit"],
    model_config: LVAEModel,
    loss_config: Optional[LVAELossConfig] = None,
    gaussian_lik_config: Optional[GaussianLikelihoodConfig] = None,
    nm_config: Optional[MultiChannelNMConfig] = None,
    nm_lik_config: Optional[NMLikelihoodConfig] = None,
    optimizer_config: Optional[OptimizerModel] = None,
    lr_scheduler_config: Optional[LrSchedulerModel] = None,
) -> VAEBasedAlgorithm:
    """Instantiate the split algorithm config.

    Parameters
    ----------
    algorithm : Literal["muspit", "denoisplit"]
        The algorithm type.
    model_config : LVAEModel
        The LVAE model configuration.
    loss_config : Optional[LVAELossConfig]
        The loss configuration. Default is None.
    gaussian_lik_config : Optional[GaussianLikelihoodConfig]
        The Gaussian likelihood configuration. Default is None.
    nm_config : Optional[MultiChannelNMConfig]
        The noise model configuration. Default is None.
    nm_lik_config : Optional[NMLikelihoodConfig]
        The noise model likelihood configuration. Default is None.
    optimizer_config : Optional[OptimizerModel]
        The optimizer configuration. Default is None.
    lr_scheduler_config : Optional[LrSchedulerModel]
        The learning rate scheduler configuration. Default is None.

    Returns
    -------
    VAEAlgorithmConfig
        The split algorithm configuration.
    """
    # TODO: VAEBasedAlgorithm in CAREamics needs to be updated so that
    #   - optimizer can be None
    #   - lr_schedular can be None

    # this is so that we fall back on the defaults set in VAEBasedAlgorithm if any
    # of the optional params are passed as none
    vae_algorithm_params = {
        "algorithm": algorithm,
        "model": model_config,
    }
    optional_params = {
        "loss": loss_config,
        "gaussian_likelihood": gaussian_lik_config,
        "noise_model": nm_config,
        "noise_model_likelihood": nm_lik_config,
        "optimizer": optimizer_config,
        "lr_scheduler": lr_scheduler_config,
    }
    for key, value in optional_params.items():
        # values which are None do not get added,
        # now defaults in VAEBasedAlgorithm will be used
        if value is not None:
            vae_algorithm_params[key] = value

    return VAEBasedAlgorithm(**vae_algorithm_params)
