from typing import Callable
from torch.utils.data import DataLoader, random_split
from src.data.dataset import FacialKeypointsDataset


def get_data_loader(data_type: str, batch_size: int, data_transform: Callable, fraction: float = None) -> DataLoader:

    dataset = FacialKeypointsDataset(csv_file=f'data/{data_type}_frames_keypoints.csv',
                                     root_dir=f'data/{data_type}/',
                                     transform=data_transform)

    if fraction is not None:
        assert 0.0 <= fraction <= 1.0, 'fraction needs to be between 0/1'
        lengths = [int(len(dataset) * fraction), (len(dataset) - int(len(dataset) * fraction))]
        dataset, _ = random_split(dataset, lengths)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
