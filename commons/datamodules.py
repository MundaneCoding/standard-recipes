import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# generic dataset
class StandardDataset(Dataset):
    def __init__(self, data_dir, csv_path, label_dir=None, *args, **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, index):
        # start with
        # curr_info = self.df.iloc[index]
        raise NotImplementedError

    

# generic data module
class StandardDatamodule(pl.LightningDataModule):
    def __init__(self, data_dir, index_dir, label_dir=None, 
                 transform=None, 
                 batch_size=1, 
                 caching=False, 
                 dataset=StandardDataset,
                 *args, **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.label_dir = label_dir
        self.transform = transform
        self.batch_size = batch_size
        self.caching = caching

        self.dataset = dataset

        self.num_worker = 1
        
    def prepare_data(self):
        return NotImplemented
    
    def setup(self, stage: str) -> None:
        if stage == 'test_only':
            self.ts_set = self.dataset(data_dir=self.data_dir,
                                       csv_path=os.path.join(self.index_dir, 'index_test.csv'),
                                       label_dir=self.label_dir,
                                       stage='test')
        else:
            self.tr_set = self.dataset(data_dir=self.data_dir,
                                       csv_path=os.path.join(self.index_dir, 'index_train.csv'),
                                       label_dir=self.label_dir,
                                       stage='train')
            self.va_set = self.dataset(data_dir=self.data_dir,
                                       csv_path=os.path.join(self.index_dir, 'index_val.csv'),
                                       label_dir=self.label_dir,
                                       stage='val')
            self.ts_set = self.dataset(data_dir=self.data_dir,
                                       csv_path=os.path.join(self.index_dir, 'index_test.csv'),
                                       label_dir=self.label_dir,
                                       stage='test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(self.tr_set)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.va_set)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.ts_set)

    def get_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_worker)
    

if __name__ == '__main__':
    datamodule = StandardDatamodule()