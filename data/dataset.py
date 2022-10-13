from PIL import Image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms as tvt


class LatexDataset(Dataset):
    def __init__(self, data_path, img_path, data_type: str, n_sample: int = None, dataset="100k"):
        super().__init__()
        assert data_type in ["train", "test", "validate"], "Not found data type"
        csv_path = data_path + f"/im2latex_{data_type}.csv"
        df = pd.read_csv(csv_path)
        if not n_sample:
            df = df.head(n_sample)
        df["image"] = df.image.map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        if dataset == "100k":
            self.transform = tvt.Compose([tvt.ToTensor(), tvt.Grayscale()])
        else:
            self.transform = tvt.Compose([tvt.ToTensor()])

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        formula = item["formula"]
        image = Image.open(item["image"])
        image = self.transform(image)

        return image, formula
