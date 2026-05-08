import random

import numpy as np
from pandas import read_csv
import torch
import matplotlib.pyplot as plt


def fix_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cast_model_floats(
    model: torch.nn.Module, dtype: torch.dtype = torch.float32
) -> torch.nn.Module:
    """
    Cast every floating parameter and buffer to the requested dtype.
    """
    model.to("cpu")
    model.to(dtype=dtype)

    incompatible_tensors = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if torch.is_floating_point(tensor) and tensor.dtype != dtype:
            incompatible_tensors.append(f"{name}:{tensor.dtype}")

    if incompatible_tensors:
        preview = ", ".join(incompatible_tensors[:5])
        raise TypeError(
            f"Model still contains floating tensors that are not {dtype}: {preview}"
        )

    return model


def get_ignored_pixels(pred: torch.Tensor) -> int:
    """Get the number of ignored pixels in the predictions.

    Some of the pixels present in the last few rows and columns of predicted images
    should be discarded. Indeed, the prediction there is always zero, since they don't
    come in batches as not multiples of the tile size.

    To spot these areas, we analyze the pixel-wise std of a predicted image. When that
    is zero, we know that the last few rows and columns are all zero.

    NOTE: this function is just to understand the number of ignored pixels.
    The function to discard these pixels is `ignore_pixels`.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted image. Shape is (H, W, C).

    Returns
    -------
    int
        Number of ignored pixels.
    """
    ignored_pixels = 1
    while pred[-ignored_pixels:, -ignored_pixels:, ...].std() == 0:
        ignored_pixels += 1
    ignored_pixels -= 1
    print(f"In {pred.shape}, last {ignored_pixels} many rows and columns are zero.")
    return ignored_pixels


def plot_probability_distribution(noise_model, signalBinIndex, histogram, channel):
    # TODO add typing
    """Plots probability distribution P(x|s) for a certain ground truth signal.

    Predictions from both Histogram and GMM-based Noise models are displayed for comparison.

    Parameters
    ----------
    signalBinIndex: int
        index of signal bin. Values go from 0 to number of bins (`n_bin`).
    histogramNoiseModel: Histogram based noise model
    gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
        Object containing trained parameters.
    device: GPU device
    """

    n_bin = 100  # TODO clarify this and signalBinIndex
    histBinSize = (
        noise_model.max_signal.item() - noise_model.min_signal.item()
    ) / n_bin
    querySignal = (
        signalBinIndex
        / float(n_bin)
        * (noise_model.max_signal - noise_model.min_signal)
        + noise_model.min_signal
    )
    querySignal += histBinSize / 2

    queryObservations = torch.arange(
        noise_model.min_signal.item(), noise_model.max_signal.item(), histBinSize
    )
    queryObservations += histBinSize / 2
    # TODO this is ugly, refactor
    noise_model.mode = "inference"
    noise_model.min_signal = torch.tensor(
        noise_model.min_signal, device=noise_model.device
    )
    noise_model.max_signal = torch.tensor(
        noise_model.max_signal, device=noise_model.device
    )
    # noise_model.tol = torch.tensor(noise_model.tol, device=noise_model.device)
    pTorch = noise_model.likelihood(
        observations=queryObservations, signals=torch.tensor(querySignal)
    )
    pNumpy = pTorch.cpu().detach().numpy()

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Noise model for channel {channel}")
    plt.subplot(1, 2, 1)
    plt.xlabel("Observation Bin")
    plt.ylabel("Signal Bin")
    plt.imshow(histogram**0.25, cmap="gray")
    plt.axhline(y=signalBinIndex + 0.5, linewidth=5, color="blue", alpha=0.5)
    plt.subplot(1, 2, 2)

    # histobs_repeated = np.repeat(histobs, 2)
    # queryObservations_repeated = np.repeat(queryObservations_numpy, 2)

    plt.plot(
        queryObservations,
        pNumpy,
        label="GMM : " + " signal = " + str(np.round(querySignal, 2)),
        marker=".",
        color="red",
        linewidth=2,
    )
    plt.xlabel("Observations (x) for signal s = " + str(querySignal))
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal))

    plt.legend()
    return {
        "gmm": {"x": queryObservations, "p": pNumpy},
    }


def plot_training_metrics(file_path: str):
    csv_file = read_csv(file_path)
    csv_file.plot(x="epoch", y=["val_loss"]) # TODO add more metrics to plot

def clean_ax(ax):
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def plot_input_patches(dataset, num_channels: int, num_samples: int = 3, samples_idxs=None, patch_size=None):
    old_patch_size = None
    if patch_size is not None:
        old_patch_size = dataset._img_sz
        grid_size = dataset._grid_sz
        dataset.set_img_sz((patch_size,patch_size), grid_size)
    
    if samples_idxs is None:
        # Select 3 random samples from the dataset
        random_samples = random.sample(range(len(dataset)), num_samples)
    
    
    input_count = dataset[0][0].shape[0]
    img_sz = 3
    # Plot all dimensions of the selected samples
    _,ax = plt.subplots(figsize=(img_sz*(input_count + num_channels), img_sz*num_samples), ncols=(input_count + num_channels), nrows=num_samples)

    for i, sample_idx in enumerate(random_samples):
        inp, sample = dataset[sample_idx]  # Get the target data of the sample

        for input_ch in range(input_count):
            ax[i, input_ch].imshow(inp[input_ch])
            if input_ch == 0:
                ax[i, input_ch].set_title(f"Primary Input")
            else:    
                ax[i, input_ch].set_title(f"Input LC[{input_ch}] ")
        # Plot each dimension
        vmin = sample.min()
        vmax = sample.max()
        for channel_idx in range(num_channels):
            ax[i, input_count+channel_idx].imshow(sample[channel_idx], vmin=vmin, vmax=vmax)
            ax[i, input_count+channel_idx].set_title(f"Channel {channel_idx + 1}")
    
    if old_patch_size is not None:
        dataset.set_img_sz((old_patch_size,old_patch_size), grid_size)
    
    clean_ax(ax)
    return random_samples


def plot_input_patches_3d(dataset, num_channels: int, num_samples: int = 3, samples_idxs=None, patch_size=None):
    old_patch_size = None
    if patch_size is not None:
        old_patch_size = dataset._img_sz
        grid_size = dataset._grid_sz
        dataset.set_img_sz((patch_size,patch_size), grid_size)
    
    if samples_idxs is None:
        # Select 3 random samples from the dataset
        random_samples = random.sample(range(len(dataset)), num_samples)    
    
    input_count = dataset[0][0].shape[0]
    img_sz = 3
    # Plot all dimensions of the selected samples
    _, ax = plt.subplots(
        figsize=(img_sz*(input_count + num_channels), img_sz*num_samples),
        ncols=(input_count + num_channels),
        nrows=num_samples
    )

    for i, sample_idx in enumerate(random_samples):
        inp, sample = dataset[sample_idx]  # Get the target data of the sample
        
        # get random slice # TODO allow selection? also other datasets might have different shapes or dimension ordering
        random_slice = random.randint(0, inp.shape[-3]-1)
        
        for input_ch in range(input_count):
            ax[i, input_ch].imshow(inp[input_ch][random_slice])
            if input_ch == 0:
                ax[i, input_ch].set_title(f"Primary Input")
            else:    
                ax[i, input_ch].set_title(f"Input LC[{input_ch}] ")
        # Plot each dimension
        for channel_idx in range(num_channels):
            ax[i, input_count+channel_idx].imshow(sample[channel_idx][random_slice])
            ax[i, input_count+channel_idx].set_title(f"Channel {channel_idx + 1}")
    
    if old_patch_size is not None:
        dataset.set_img_sz((old_patch_size,old_patch_size), grid_size)
    
    clean_ax(ax)
    return random_samples


def plot_training_outputs(dataset, model, num_channels: int, num_samples: int = 3):
    # Select 3 random samples from the dataset
    random_samples = random.sample(range(len(dataset)), num_samples)
    # Plot all channels of the selected samples
    fig, axs = plt.subplots(num_channels, num_samples*2, figsize=(num_channels*4, 12))
    fig.suptitle("Random Sample Predictions - All Channels")
    for i, sample_idx in enumerate(random_samples):
        sample, target = dataset[sample_idx]  # Get the input data of the sample
        num_channels, _, _ = target.shape
        # Predict using the model
        model.eval()
        model.cuda()
        with torch.no_grad():
            prediction, _ = model(torch.tensor(sample).unsqueeze(0).cuda())  # Assuming the model takes input of shape (batch_size, num_channels, height, width)
        # Plot each channel
        for channel_idx in range(num_channels):
            axs[i, channel_idx].imshow(target[channel_idx], cmap='gray')
            axs[i, channel_idx].set_title(f"Idx {sample_idx} - Channel {channel_idx}")
            axs[i, channel_idx].axis('off')
            axs[i, channel_idx + num_channels].imshow(prediction.squeeze()[channel_idx].cpu(), cmap='gray')
            axs[i, channel_idx + num_channels].set_title(f"Idx {sample_idx} - Prediction {channel_idx}")
            axs[i, channel_idx + num_channels].axis('off')
    plt.tight_layout()
    plt.show()
    

def plot_individual_samples(stitched_samples):
    assert len(stitched_samples) > 1, "At least 2 samples are required to plot"
    num_channels = stitched_samples[0].shape[-1]
    num_samples = len(stitched_samples)
    _, ax = plt.subplots(num_channels, num_samples, figsize=(30, 20)) # TODO make figsize dynamic
    for i, sample in enumerate(stitched_samples):
        for j in range(num_channels):
            ax[j, i].imshow(sample[..., j].squeeze(), cmap='gray')
            ax[j, i].axis('off')
            ax[j, i].set_title(f"Sample {i} - Channel {j + 1}")
    plt.tight_layout()
    plt.show()