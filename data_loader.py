# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import h5py
# from pathlib import Path


# def normalize_point_cloud(pc):
#     """
#     Normalize point cloud to zero mean and unit sphere.
#     Args:
#         pc: numpy array (N, 3)
#     Returns:
#         normalized pc (N, 3)
#     """
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     furthest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / furthest_distance
#     return pc


# def random_jitter(pc, sigma=0.01, clip=0.02):
#     """Add random jitter to each point."""
#     jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
#     return pc + jitter


# def random_rotate(pc):
#     """Random rotation around the up-axis (y-axis)."""
#     theta = np.random.uniform(0, 2 * np.pi)
#     rot_matrix = np.array([
#         [np.cos(theta), 0, np.sin(theta)],
#         [0, 1, 0],
#         [-np.sin(theta), 0, np.cos(theta)]
#     ])
#     return pc @ rot_matrix.T


# class ModelNet40Dataset(Dataset):
#     def __init__(self, root, split='train', npoints=1024, augment=True, use_gp_pcs=False, gp_sampler=None):
#         """
#         Args:
#             root: root folder containing ModelNet40 .h5 files
#             split: 'train' or 'test'
#             npoints: number of points to sample per shape
#             augment: whether to apply augmentations
#             use_gp_pcs: if True, apply GP-PCS sampler
#             gp_sampler: callable sampler (if use_gp_pcs=True)
#         """
#         self.root = Path(root)
#         self.split = split
#         self.npoints = npoints
#         self.augment = augment
#         self.use_gp_pcs = use_gp_pcs
#         self.gp_sampler = gp_sampler

#         # Load data files
#         self.data, self.labels = self._load_data()

#     def _load_data(self):
#         # ModelNet40 dataset is often stored in h5 format
#         all_data, all_labels = [], []
#         file_list = sorted(self.root.glob(f'{self.split}*.h5'))
#         if len(file_list) == 0:
#             print(f"[ERROR] No .h5 files found for split='{self.split}' in {self.root}")
#             print("Expected something like train0.h5, train1.h5, ... or test0.h5, test1.h5")
#             raise FileNotFoundError(f"No .h5 files found in {self.root}")
#         else:
#             print(f"[INFO] Found {len(file_list)} files for split='{self.split}':")
#             for f in file_list:
#                 print(f"   {f.name}")


#         for filename in file_list:
#             with h5py.File(filename, 'r') as f:
#                 data = f['data'][:].astype('float32')
#                 label = f['label'][:].astype('int64')
#                 all_data.append(data)
#                 all_labels.append(label)

#         all_data = np.concatenate(all_data, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
#         return all_data, all_labels

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         point_set = self.data[idx][:self.npoints]
#         label = self.labels[idx]

#         # Normalize
#         point_set = normalize_point_cloud(point_set)

#         # Optional GP-PCS sampling (later)
#         if self.use_gp_pcs and self.gp_sampler is not None:
#             point_set = self.gp_sampler(point_set)

#         # Data augmentations
#         if self.augment and self.split == 'train':
#             point_set = random_rotate(point_set)
#             point_set = random_jitter(point_set)

#         # Convert to torch tensors
#         point_set = torch.from_numpy(point_set).float()
#         label = torch.tensor(label).long()

#         return point_set, label


# def get_dataloader(root, batch_size=32, split='train', npoints=1024, augment=True,
#                    use_gp_pcs=False, gp_sampler=None, num_workers=4):
#     dataset = ModelNet40Dataset(
#         root=root,
#         split=split,
#         npoints=npoints,
#         augment=augment,
#         use_gp_pcs=use_gp_pcs,
#         gp_sampler=gp_sampler
#     )

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(split == 'train'),
#         num_workers=num_workers,
#         drop_last=True
#     )
#     return dataloader

# data_loader.py
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    furthest_distance = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if furthest_distance > 0:
        pc = pc / furthest_distance
    return pc

def random_jitter(pc, sigma=0.01, clip=0.02):
    jitter = np.clip(sigma * np.random.randn(*pc.shape), -clip, clip)
    return pc + jitter

def random_rotate(pc):
    theta = np.random.uniform(0, 2 * np.pi)
    rot = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return pc @ rot.T

class ModelNet40Dataset(Dataset):
    def __init__(self, root, split='train', npoints=1024, augment=True, use_gp_pcs=False, gp_sampler=None):
        self.root = Path(root)
        self.split = split
        self.npoints = npoints
        self.augment = augment
        self.use_gp_pcs = use_gp_pcs
        self.gp_sampler = gp_sampler

        self.data, self.labels = self._load_data()

    def _load_data(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        pattern = f"{self.split}*.h5"
        files = sorted(self.root.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No files matching {pattern} in {self.root}")
        all_data = []
        all_labels = []
        for fpath in files:
            with h5py.File(fpath, 'r') as f:
                # typical keys: 'data', 'label'
                if 'data' in f and 'label' in f:
                    d = f['data'][:].astype('float32')
                    l = f['label'][:].astype('int64')
                else:
                    # fallback heuristics
                    keys = list(f.keys())
                    if len(keys) >= 2:
                        d = f[keys[0]][:].astype('float32')
                        l = f[keys[1]][:].astype('int64')
                    else:
                        raise RuntimeError(f"Unexpected h5 file content: {fpath}, keys={list(f.keys())}")
                all_data.append(d)
                all_labels.append(l)
        data = np.concatenate(all_data, axis=0)
        labels = np.concatenate(all_labels, axis=0).reshape(-1)
        return data, labels

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        pts = self.data[idx]
        # ensure N >= npoints
        if pts.shape[0] >= self.npoints:
            pts = pts[:self.npoints]
        else:
            repeats = int(np.ceil(self.npoints / pts.shape[0]))
            pts = np.tile(pts, (repeats, 1))[:self.npoints]
        pts = normalize_point_cloud(pts)
        if self.augment and self.split == 'train':
            pts = random_rotate(pts)
            pts = random_jitter(pts)
        if self.use_gp_pcs and self.gp_sampler is not None:
            pts = self.gp_sampler(pts)
        pts = torch.from_numpy(pts).float()
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return pts, label

def get_dataloader(root, batch_size=8, split='train', npoints=1024, augment=True, use_gp_pcs=False, gp_sampler=None, num_workers=0):
    ds = ModelNet40Dataset(root=root, split=split, npoints=npoints, augment=augment, use_gp_pcs=use_gp_pcs, gp_sampler=gp_sampler)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers, drop_last=False)
    return dl
