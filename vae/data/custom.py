from torch.utils.data import Dataset
import os
from vae.data.base import ImagePaths


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, size, data_root, training_images_list_file, **kwargs):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        paths = [os.path.join(data_root, p) for p in paths]
        # self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.data = ImagePaths(paths=paths, size=size, random_crop=True, **kwargs) 


class CustomTest(CustomBase):
    def __init__(self, size, data_root, test_images_list_file, **kwargs):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        paths = [os.path.join(data_root, p) for p in paths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, **kwargs)
