from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

##### argument parser
class StandardArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        ### dev
        self.add_argument('--debug', action='store_true', default=False)

        ### model and training
        self.add_argument('--batch_size', default=64, type=int)
        self.add_argument('--max_epochs', default=20, type=int)
        self.add_argument('--resume', default=None, type=str)

        ### telemetry
        self.add_argument('--log_dir', default='./log', type=str)

        ### hardware
        self.add_argument('--gpus', default=[0], nargs='+', type=int)

    def parse_args(self):
        args = super().parse_args()
        if args.gpus[0] == -1: args.gpus = -1
        return args
    

##### generic trainer
class StandardTrainer(pl.Trainer):
    def __init__(self, sargs, val_check_interval=0.25, *args, **kwargs):
        ## checkpointing behaviors
        chpt_cb_loss = ModelCheckpoint(
            monitor='val_loss_epoch',
            mode='min',
            verbose=True,
            auto_insert_metric_name=True,
            save_on_train_epoch_end=False,
            save_top_k=3,
            filename='{epoch}-{step}-{val_loss_epoch:.5f}'
        )

        chpt_cb_epoch = ModelCheckpoint(
            monitor='epoch',
            mode='max',
            save_on_train_epoch_end=True,
            save_top_k=3,
            filename='chpt-{epoch:02d}'
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        super().__init__(accelerator='cuda', 
                         devices=sargs.gpus, 
                         max_epochs=sargs.max_epochs, 
                         val_check_interval=val_check_interval, 
                         default_root_dir=sargs.log_dir, 
                         enable_progress_bar=True, 
                         callbacks=[chpt_cb_loss, chpt_cb_epoch, lr_monitor], 
                         strategy="ddp_find_unused_parameters_false",
                         fast_dev_run=sargs.debug, 
                         log_every_n_steps=1,
                         *args, **kwargs)
        

if __name__ == '__main__':
    parser = StandardArgParser()
    args = parser.parse_args()
    print(args)