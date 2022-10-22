from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import torchvision
from torchvision import transforms as tvt
import math

class LatexDataset(Dataset):
    def __init__(
        self, data_path, img_path, data_type: str, n_sample: int = None, dataset="100k"
    ):
        super().__init__()
        assert data_type in ["train", "test", "validate"], "Not found data type"
        csv_path = data_path + f"/im2latex_{data_type}.csv"
        df = pd.read_csv(csv_path)
        if n_sample:
            df = df.head(n_sample)
        df["image"] = df.image.map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        formula = item["formula"]
        image = torchvision.io.read_image(item["image"])
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # transform image to [-1, 1]
        return image, formula

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
