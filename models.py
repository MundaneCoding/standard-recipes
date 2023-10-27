from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import pytorch_lightning as pl


class StandardModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def save_config(self):
        self.save_hyperparameters()

    def _get_optimizer(self):
        raise NotImplementedError
    
    def _get_lr_scheduler(self, optimizer):
        raise NotImplementedError
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self._get_optimizer()
        lr_scheduler = self._get_lr_scheduler(optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
    
    ### training
    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)
    
    ### validation
    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().validation_step(*args, **kwargs)
    
    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()
    
    ### forward step
    def _forward_step(self, batch, batch_idx):
        raise NotImplementedError