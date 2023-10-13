import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# generic dataset
class StandardDataset(Dataset):
    def __init__(self, data_dir, csv_path) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, index):
        # start with
        # curr_info = self.df.iloc[index]
        raise NotImplementedError

    

# generic data module
class StandardDatamodule(pl.LightningDataModule):
    def __init__(self, data_dir, index_dir, transform=None, batch_size=1, caching=False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.transform = transform
        self.batch_size = batch_size
        self.caching = caching

        
    def prepare_data(self):
        return NotImplemented
    
    def setup(self, stage: str) -> None:
        if stage == 'test_only':
            self.ts_set = StandardDataset(data_dir=self.data_dir,
                                        csv_path=os.path.join(self.index_dir, 'index_test.csv'))
        else:
            self.tr_set = StandardDataset(data_dir=self.data_dir,
                                        csv_path=os.path.join(self.index_dir, 'index_train.csv'))
            self.va_set = StandardDataset(data_dir=self.data_dir,
                                        csv_path=os.path.join(self.index_dir, 'index_val.csv'))
            self.ts_set = StandardDataset(data_dir=self.data_dir,
                                        csv_path=os.path.join(self.index_dir, 'index_test.csv'))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(self.tr_set)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.va_set)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.ts_set)

    def get_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size)
    

if __name__ == '__main__':
    datamodule = StandardDatamodule()