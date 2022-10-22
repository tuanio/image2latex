from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torchvision
from torchvision import transforms as tvt


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
