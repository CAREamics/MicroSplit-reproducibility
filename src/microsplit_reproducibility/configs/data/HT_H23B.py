from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType
from typing import Optional

class HTH23BConfig(DatasetConfig):
    channel_list: Optional[list[str]] = ["puncta", "foreground"]
    background_values: list[int] = [0, 0]
    test_frame_idx: int = 8  # Default test frame index


def get_data_configs(test_frame_idx=8) -> tuple[HTH23BConfig, HTH23BConfig, HTH23BConfig]:
    train_data_config = HTH23BConfig(
        data_type=DataType.HTH23BData,
        datasplit_type=DataSplitType.Train,
        image_size=(64, 64),
        grid_size=32,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=6675,
        input_has_dependant_noise=True,
        multiscale_lowres_count=1,
        use_one_mu_std=True,
        train_aug_rotate=True,
        target_separate_normalization=True,
        uncorrelated_channels=True,
        uncorrelated_channel_probab=1,
        input_is_sum=True,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
    )

    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    test_data_config = val_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Test,
            test_frame_idx=test_frame_idx,
            uncorrelated_channels=False,  # Test data is already channel-mixed
        )
    )
    return train_data_config, val_data_config, test_data_config
