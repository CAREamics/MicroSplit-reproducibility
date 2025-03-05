import os

import numpy as np

import nd2

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples


def get_ms_based_datafiles(ms: str):
    return [f"Set{i}/uSplit_{ms}.nd2" for i in range(1, 7)]


def get_raw_files_dict():
    files_dict = {
        "high": [
            "uSplit_14022025_highSNR.nd2",
            "uSplit_20022025_highSNR.nd2",
            "uSplit_20022025_001_highSNR.nd2",
        ],
        "mid": [
            "uSplit_14022025_midSNR.nd2",
            "uSplit_20022025_midSNR.nd2",
            "uSplit_20022025_001_midSNR.nd2",
        ],
        "low": [
            "uSplit_14022025_lowSNR.nd2",
            "uSplit_20022025_lowSNR.nd2",
            "uSplit_20022025_001_lowSNR.nd2",
        ],
        "verylow": [
            "uSplit_14022025_verylowSNR.nd2",
            "uSplit_20022025_verylowSNR.nd2",
            "uSplit_20022025_001_verylowSNR.nd2",
        ],
        "2ms": get_ms_based_datafiles("2ms"),
        "3ms": get_ms_based_datafiles("3ms"),
        "5ms": get_ms_based_datafiles("5ms"),
        "20ms": get_ms_based_datafiles("20ms"),
        "500ms": get_ms_based_datafiles("500ms"),
    }
    # check that the order is correct
    keys = ["high", "mid", "low", "verylow"]
    for key1 in keys:
        filetokens1 = list(map(lambda x: x.replace(key1, ""), files_dict[key1]))
        for key2 in keys:
            filetokens2 = list(map(lambda x: x.replace(key2, ""), files_dict[key2]))
            assert np.array_equal(
                filetokens1, filetokens2
            ), f"File tokens are not equal for {key1} and {key2}"
    return files_dict


def load_one_fpath(fpath, channel_list):
    with nd2.ND2File(fpath) as nd2file:
        # axes = "".join(nd2_file.sizes.keys())  # PCYX [4 of possible 7 "TPZCYXS"]
        data = nd2file.asarray()                 # shape: (20, 19, 1608, 1608)
    # Here, 20 are different locations and 19 are different channels.
    data = data[:, channel_list, ...]
    # swap the second and fourth axis
    data = np.swapaxes(data[..., None], 1, 4)[:, 0]

    fname_prefix = "_".join(os.path.basename(fpath).split(".")[0].split("_")[:-1])
    if fname_prefix == "uSplit_20022025_001":
        data = np.delete(data, 2, axis=0)
    elif fname_prefix == "uSplit_14022025":
        data = np.delete(data, [17, 19], axis=0)

    # shape: (20, 1608, 1608, C)
    return data


def load_data(datadir, channel_list, dataset_type):
    files_dict = get_raw_files_dict()[dataset_type]
    data_list = []
    for fname in files_dict:
        fpath = os.path.join(datadir, fname)
        data = load_one_fpath(fpath, channel_list)
        data_list.append(data)
    if len(data_list) > 1:
        data = np.concatenate(data_list, axis=0)
    else:
        data = data_list[0]
    return data


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    data = load_data(
        datadir,
        channel_list=data_config.channel_idx_list,
        dataset_type=data_config.dset_type,
    )
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float32)
    else:
        raise Exception("invalid datasplit")

    return data
