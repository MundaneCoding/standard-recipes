from commons.datamodules import StandardDatamodule, StandardDataset
import torchvision.transforms as tf
from PIL import Image

class SimpleDatamodule(StandardDatamodule):
    def __init__(self, data_dir, csv_path, label_dir, *arg, **kwargs) -> None:
        super().__init__(data_dir, csv_path, label_dir, *arg, **kwargs)

        self.resize = tf.Resize((self.image_size, self.image_size))
        self.transform = tf.Compose([tf.PILToTensor(), self.resize])

    def __getitem__(self, index):
        curr_info = self.df.iloc[index]

        x = Image.open(curr_info['image_path'])
        x = self.transform(x)
        y = curr_info['ground_truth']

        return {'x': x, 'y': y}