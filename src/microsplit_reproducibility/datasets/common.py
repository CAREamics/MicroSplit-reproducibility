from typing import Callable, Union

import torch
from numpy.typing import NDArray

from careamics.lvae_training.dataset import DatasetConfig, DataType, DataSplitType
from careamics.lvae_training.dataset import (
    LCMultiChDloader,
    MultiChDloader,
    # MultiChDloaderRef,
    MultiFileDset,
    MultiCropDset
)

from .HT_H23B import MultiChDloaderRef


SplittingDataset = Union[LCMultiChDloader, MultiChDloader, MultiFileDset, MultiCropDset]


def create_train_val_datasets(
    datapath: str,
    train_config: DatasetConfig,
    val_config: DatasetConfig,
    test_config: DatasetConfig,
    load_data_func: Callable[..., NDArray],
) -> tuple[SplittingDataset, SplittingDataset, SplittingDataset, tuple[float, float]]:
    if train_config.data_type in [
        DataType.TavernaSox2Golgi,
        DataType.Dao3Channel,
        DataType.Dao3ChannelWithInput,
        # DataType.ExpMicroscopyV1,
        DataType.ExpMicroscopyV2,
        DataType.TavernaSox2GolgiV2,
    ]:
        dataset_class = MultiFileDset
    elif train_config.multiscale_lowres_count > 1:
        dataset_class = LCMultiChDloader
    elif train_config.data_type in [
        DataType.HTH23BData]:
        dataset_class = MultiChDloaderRef
    else:
        dataset_class = MultiChDloader

    train_data = dataset_class(
        train_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    max_val = train_data.get_max_val()
    val_config.max_val = max_val
    if train_config.datasplit_type == DataSplitType.All:
        val_config.datasplit_type = DataSplitType.All
        test_config.datasplit_type = DataSplitType.All # TODO temporary hack
    val_data = dataset_class(
        val_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    test_config.max_val = max_val
    test_data = dataset_class(
        test_config,
        datapath,
        load_data_fn=load_data_func,
        val_fraction=0.1,
        test_fraction=0.1,
    )
    mean_val, std_val = train_data.compute_mean_std()
    train_data.set_mean_std(mean_val, std_val)
    val_data.set_mean_std(mean_val, std_val)
    test_data.set_mean_std(mean_val, std_val)
    data_stats = train_data.get_mean_std()

    # NOTE: "input" mean & std are computed over the entire dataset and repeated for each channel.
    # On the contrary, "target" mean & std are computed separately for each channel.
    # manipulate data stats to only have one mean and std for the target
    assert isinstance(data_stats, tuple)
    assert isinstance(data_stats[0], dict)

    data_stats = (
        torch.tensor(data_stats[0]["target"]),
        torch.tensor(data_stats[1]["target"]),
    )

    return train_data, val_data, test_data, data_stats


def get_target_images(test_dset: SplittingDataset) -> NDArray:
    """Get the target images."""
    if test_dset.data_type in [
        DataType.HTIba1Ki67,
    ]:
        return test_dset._data
