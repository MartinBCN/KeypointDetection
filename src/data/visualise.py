import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor


def show_keypoints(image: np.array, key_pts: np.array, ax: Axes = None) -> None:
    """
    Show image with keypoints

    Parameters
    ----------
    image
    key_pts
    ax

    Returns
    -------

    """
    if ax is None:
        ax = plt
    ax.imshow(image)
    ax.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


def visualise_batch(image: Tensor, keypoints: Tensor) -> None:
    """
    Visualise a batch from the dataloader

    Parameters
    ----------
    image: Tensor
        Format is [n, 1, 224, 224] (the 224 is in principle arbirtrary)
    keypoints: Tensor
        Format is [n, 68, 2] (the 68 is in principle arbitrary)

    Returns
    -------
    None
    """
    n = image.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(20, 10))

    for i in range(n):
        img = image[i, 0, :, :].detach().cpu().numpy()
        kp = keypoints[i, :, :].detach().cpu().numpy()
        kp = kp * 50 + 100
        show_keypoints(img, kp, axes[i])

    plt.show()
