from commons.models import StandardModel
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

class SimpleClassifier(StandardModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 28)
        self.fc2 = nn.Linear(28, 84)
        self.fc3 = nn.Linear(84, 10)

        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.loss = torch.nn.MSELoss()

    def _get_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def _get_lr_scheduler(self, optimizer):
        return {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer), 
            'monitor': 'val_loss_epoch'
        }
    
    ### training
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, pred = self._forward_step(batch, batch_idx)
        self.log('train_loss', loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    ### validation
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        with torch.no_grad():
            loss, pred = self._forward_step(batch, batch_idx)
            self.log('val_loss', loss, sync_dist=True, on_step=True, on_epoch=True)

        return loss

    ### forward step
    def _forward_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']

        # forward calls
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = self.fc3(x)

        # calculate loss
        loss = self.loss(y_hat, y)

        return y_hat, loss
    