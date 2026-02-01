from pathlib import Path
from PIL import Image
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from diffusion.data.builder import DATASETS
from einops import rearrange
import torch.nn.functional as F

@DATASETS.register_module()
class PanCancerControlNetData(Dataset):
    """
    Dataset for PixCell-256 ControlNet training.
    Returns: (vae_feat, ssl_feat, cell_mask, data_info)
    
    According to the paper:
    - Uses CellViT-SAM-H trained on 20× (0.5 μm/px) images
    - 10,000 images sampled from PanCan-30M
    - Binary cell masks extracted
    - Training with (image, UNI embedding, mask) triplets
    """
    def __init__(self, root, resolution, **kwargs):
        """
        Initialize the PanCancerControlNetData dataset.

        Args:
            root (str): Root directory of the dataset.
            resolution (int): Resolution of the images (should be 256 for ControlNet).
            **kwargs: Additional keyword arguments.
        """
        self.root = Path(root)
        self.load_vae_feat = True
        self.resolution = resolution
        self.vae_prefix = kwargs.get("vae_prefix", "sd3_vae")
        self.ssl_prefix = kwargs.get("ssl_prefix", "uni")
        self.mask_prefix = kwargs.get("mask_prefix", "cellvit_mask")  # Cell mask prefix
        self.return_img = kwargs.get("return_img", False)
        self.train_subset_keys = kwargs.get("train_subset_keys", None)

        patch_names_file = kwargs.get("patch_names_file", "patch_names_controlnet.hdf5")

        self.image_list_h5 = self.root / f"patches/metadata/{patch_names_file}"
        self.features_dir = self.root / "features"
        self.masks_dir = self.root / "masks"  # Directory for cell masks

        # Load metadata from the HDF5 file
        self._load_metadata()

        self.dummy_data = False

    def _load_metadata(self):
        """Load dataset keys and lengths from the HDF5 file."""
        self.h5 = None
        lengths = {}
        with h5py.File(self.root / self.image_list_h5, "r") as h5:
            # Filter keys based on the resolution
            keys_all = list(h5.keys())
            if self.train_subset_keys is not None:
                keys_all = [key for key in keys_all if key in self.train_subset_keys]

            self.keys = [item for item in keys_all if f"_{self.resolution}" in item]

            # Store the length of each key
            for key in self.keys:
                lengths[key] = len(h5[key])

        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = np.concatenate([[0], np.cumsum(list(lengths.values()))])

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return int(self.cumulative_lengths[-1])

    @staticmethod
    def _vae_feat_loader(path):
        """Load VAE features from a .npy file."""
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(
            mean.shape, generator=None, device=mean.device, dtype=mean.dtype
        )
        return mean + std * sample
    
    @staticmethod
    def _mask_loader(path):
        """Load cell mask from a .npy or .png file."""
        if path.suffix == '.npy':
            mask = torch.from_numpy(np.load(path))
        else:
            # Load as binary image
            mask = Image.open(path).convert('L')
            mask = np.array(mask)
            mask = torch.from_numpy(mask > 0).float()
        
        # Ensure mask is [1, H, W]
        if mask.ndim == 1:
            # Handle flattened mask - need to reshape
            # Assuming square mask
            size = int(np.sqrt(mask.numel()))
            mask = mask.reshape(size, size)
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = mask[0:1]  # Take first channel if multiple
        
        return mask
    
    def load_from_image_name(self, img_name):
        """
        Load VAE features, SSL embeddings, and cell mask for a given image.
        
        Args:
            img_name (str): Name of the image file.
            
        Returns:
            tuple: (vae_feat, ssl_feat, cell_mask, data_info)
        """
        # Load VAE and SSL features
        vae_path = img_name.replace(".jpeg", f"_{self.vae_prefix}.npy")
        ssl_path = img_name.replace(".jpeg", f"_{self.ssl_prefix}.npy")
        mask_path = img_name.replace(".jpeg", f"_{self.mask_prefix}.npy")
        
        vae_path = img_name.replace(".png", f"_{self.vae_prefix}.npy")
        ssl_path = img_name.replace(".png", f"_{self.ssl_prefix}.npy")
        mask_path = img_name.replace(".png", f"_{self.mask_prefix}.png")

        vae_feat = self._vae_feat_loader(self.features_dir / vae_path)
        # Add this line to remove the extra batch dimension:
        if vae_feat.ndim == 4 and vae_feat.shape[0] == 1:
            vae_feat = vae_feat.squeeze(0)  # (1, 16, 32, 32) -> (16, 32, 32)

        lt_sz = self.resolution // 8
        assert vae_feat.shape == (16, lt_sz, lt_sz)
        ssl_feat = torch.from_numpy(np.load(self.features_dir / ssl_path))
        cell_mask = self._mask_loader(self.masks_dir / mask_path)
        cell_mask = cell_mask.unsqueeze(0) # (c, h, w) -> (1, c, h, w)
        cell_mask = F.interpolate(
            cell_mask, 
            size=(self.resolution, self.resolution), 
            mode='nearest'
        )
        cell_mask = cell_mask.squeeze(0)
        lt_sz = self.resolution // 8
        assert vae_feat.shape == (16, lt_sz, lt_sz), f"Expected VAE shape (16, {lt_sz}, {lt_sz}), got {vae_feat.shape}"
        assert cell_mask.shape[-2:] == (self.resolution, self.resolution), \
            f"Expected mask shape (..., {self.resolution}, {self.resolution}), got {cell_mask.shape}"

        data_info = {
            "img_hw": torch.tensor([self.resolution] * 2, dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
            "mask_type": "binary",
            "img_name": img_name
        }

        if self.return_img:
            img = Image.open(self.root / f"patches/{img_name}")
            img = np.array(img)
            return vae_feat, ssl_feat, cell_mask, img
        return vae_feat, ssl_feat, cell_mask, data_info

    def get_data(self, idx):
        """
        Get data for a given index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: VAE features, SSL features, cell_mask, and additional data info.
        """
        if self.h5 is None:
            self.h5 = h5py.File(self.root / self.image_list_h5, "r")
        key_idx = (
            np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        )
        key = self.keys[key_idx]
        idx = idx - self.cumulative_lengths[key_idx]
        img_name = self.h5[key][idx].decode("utf-8")

        if self.dummy_data:
            return img_name

        return self.load_from_image_name(img_name)
    
    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: VAE features, SSL features, cell_mask, and additional data info.
        """
        retries = 3
        for _ in range(retries):
            try:
                return self.get_data(idx)
            except Exception as e:
                print(f"Failed to load data for index {idx}: {e}. Retrying...")
                idx = np.random.randint(len(self))

        raise RuntimeError(f"Failed to load data after {retries} attempts.")
