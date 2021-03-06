import os
from typing import Callable
from torch.utils.data import DataLoader, random_split
from src.data.dataset import FacialKeypointsDataset


def get_data_loader(data_type: str, batch_size: int, data_transform: Callable, fraction: float = None) -> DataLoader:
    """
    Facade for getting the DataLoader

    Parameters
    ----------
    data_type
    batch_size
    data_transform
    fraction

    Returns
    -------

    """
    subdirectories = {'train': 'training', 'validation': 'training', 'test': 'test'}
    subdirectory = subdirectories[data_type]

    data_path = os.environ.get('DATA_PATH', 'data')
    dataset = FacialKeypointsDataset(csv_file=f'{data_path}/{subdirectory}_frames_keypoints.csv',
                                     root_dir=f'{data_path}/{subdirectory}/',
                                     transform=data_transform)

    if data_type in ['train', 'validation']:
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        lengths = [train_size, val_size]
        datasets = dict()
        datasets['train'], datasets['validation'] = random_split(dataset, lengths)
        dataset = datasets[data_type]

    if fraction is not None:
        assert 0.0 <= fraction <= 1.0, 'fraction needs to be between 0/1'
        lengths = [int(len(dataset) * fraction), (len(dataset) - int(len(dataset) * fraction))]
        dataset, _ = random_split(dataset, lengths)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
