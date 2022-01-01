from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
import torch

import logging
from os import listdir
from os.path import splitext
from pathlib import Path

class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', transform = None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255
            
        if is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])


        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        img = np.pad(img, ((0,0),(0,1),(0,0)), 'constant') # 1 pixel padding xd                                                                                                                               
        mask = np.pad(mask, ((0,1),(0,0)), 'constant') # 1 pixel padding xd                                                                                                                                   

        if self.transform != None:
            img_aug, mask_aug = self.transform((img, mask))


        else:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'filename': str(mask_file[0]).split("/")[-1]
            }

        if img.flatten().tolist() == img_aug.flatten().tolist():
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'image_aug': torch.zeros(torch.as_tensor(img.copy()).float().contiguous().shape).long(),
                'mask_aug': torch.zeros(torch.as_tensor(mask.copy()).long().contiguous().shape).long(),
                'filename': str(mask_file[0]).split("/")[-1]
            }
        else:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'image_aug': torch.as_tensor(img_aug.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'mask_aug': torch.as_tensor(mask_aug.copy()).long().contiguous(),
                'filename': str(mask_file[0]).split("/")[-1]
            }


class BinaryDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255
            
        if is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])


        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        img = np.pad(img, ((0,0),(0,1),(0,0)), 'constant') # 1 pixel padding xd                                                                                                                               
        mask = np.pad(mask, ((0,1),(0,0)), 'constant') # 1 pixel padding xd
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'filename': str(mask_file[0]).split("/")[-1]
        }
        

def get_dataloaders(transform, paths, batch_size):
    dir_img = Path(paths[0])
    dir_mask = Path(paths[1])

    test_dir_img = Path(paths[2])
    test_dir_mask = Path(paths[3])

    dataset = SegmentationDataset(dir_img, dir_mask, 1., transform=transform)
    test_dataset = SegmentationDataset(test_dir_img, test_dir_mask, 1.)

    train_dataloader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size = batch_size)
    return train_dataloader, test_dataloader

def get_binary_dataloaders(paths, batch_size):
    dir_img = Path(paths[0])
    dir_mask = Path(paths[1])

    test_dir_img = Path(paths[2])
    test_dir_mask = Path(paths[3])

    dataset = SegmentationDataset(dir_img, dir_mask, 1.)
    test_dataset = SegmentationDataset(test_dir_img, test_dir_mask, 1.)
    
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size = batch_size)
    return train_dataloader, test_dataloader

    

    
