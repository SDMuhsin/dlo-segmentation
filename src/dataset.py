"""
PyTorch Dataset for PointWire data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class PointWireDataset(Dataset):
    """
    PointWire dataset for point cloud segmentation.

    Classes:
        0: Wire (main conductor)
        1: Endpoint (end of wire segment)
        2: Bifurcation (Y-junction)
        3: Connector (coupling points)
        4: Noise (background/outliers)
    """

    # Class distribution for weighted loss
    CLASS_WEIGHTS = torch.tensor([0.29, 2.78, 0.93, 0.76, 6.25])  # Inverse frequency based

    def __init__(self, data_path, split='train', num_points=2048, augment=False):
        """
        Args:
            data_path: Path to the dataset root (e.g., ./data/set2)
            split: 'train', 'val', or 'test'
            num_points: Number of points per sample (default 2048)
            augment: Whether to apply data augmentation
        """
        self.data_path = Path(data_path)
        self.split = split
        self.num_points = num_points
        self.augment = augment

        # Define split ranges
        if split == 'train':
            self.set_ids = list(range(0, 32))
        elif split == 'val':
            self.set_ids = list(range(32, 36))
        elif split == 'test':
            self.set_ids = list(range(36, 40))
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples_per_set = 300

        # Build sample list with validation
        self.samples = []
        self.num_points = num_points
        skipped = 0
        for set_id in self.set_ids:
            set_dir = self.data_path / f"{set_id:03d}"
            if set_dir.exists():
                for sample_id in range(self.samples_per_set):
                    pcl_path = set_dir / f"pointclouds_normed_{num_points}" / f"pcl_{sample_id:04d}.npy"
                    seg_path = set_dir / f"segmentation_normed_{num_points}" / f"seg_{sample_id:04d}.npy"
                    if pcl_path.exists() and seg_path.exists():
                        # Validate file shapes
                        try:
                            pcl = np.load(pcl_path)
                            seg = np.load(seg_path)
                            if pcl.shape == (num_points, 3) and seg.shape == (num_points,):
                                self.samples.append((pcl_path, seg_path))
                            else:
                                skipped += 1
                        except Exception:
                            skipped += 1

        print(f"Loaded {len(self.samples)} samples for {split} split from {len(self.set_ids)} sets (skipped {skipped} invalid files)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pcl_path, seg_path = self.samples[idx]

        # Load data
        pcl = np.load(pcl_path).astype(np.float32)  # (N, 3)
        seg = np.load(seg_path).astype(np.int64)    # (N,)

        # Apply augmentation
        if self.augment:
            pcl = self._augment(pcl)

        # Convert to tensors
        pcl = torch.from_numpy(pcl).transpose(0, 1)  # (3, N)
        seg = torch.from_numpy(seg)                   # (N,)

        return pcl, seg

    def _augment(self, pcl):
        """Apply random augmentations."""
        # Random rotation around Z axis
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        pcl = pcl @ rot_matrix.T

        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        pcl = pcl * scale

        # Random jittering
        noise = np.random.normal(0, 0.01, pcl.shape).astype(np.float32)
        pcl = pcl + noise

        return pcl

    @staticmethod
    def get_class_names():
        return ['Wire', 'Endpoint', 'Bifurcation', 'Connector', 'Noise']


def create_dataloaders(data_path, batch_size=16, num_workers=4, num_points=2048):
    """Create train, val, and test dataloaders."""
    train_dataset = PointWireDataset(data_path, split='train', num_points=num_points, augment=True)
    val_dataset = PointWireDataset(data_path, split='val', num_points=num_points, augment=False)
    test_dataset = PointWireDataset(data_path, split='test', num_points=num_points, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    data_path = Path("/workspace/cdlo-datasets/data/set2")

    for split in ['train', 'val', 'test']:
        dataset = PointWireDataset(data_path, split=split)
        if len(dataset) > 0:
            pcl, seg = dataset[0]
            print(f"{split}: {len(dataset)} samples, pcl shape: {pcl.shape}, seg shape: {seg.shape}")
            print(f"  Seg unique values: {seg.unique().tolist()}")
