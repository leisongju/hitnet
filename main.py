import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args
import torch
torch.set_float32_matmul_precision('high')

def load_callbacks(args):
    callbacks = [
        plc.EarlyStopping(
            monitor='loss',
            mode='min',
            patience=10,
            min_delta=0.001
        ),
        plc.ModelCheckpoint(
            monitor='loss',
            filename='best-{epoch:02d}-{val_acc:.3f}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
    ]

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))

    if load_path is None:
        model = MInterface(**vars(args))
    else:
        model = MInterface.load_from_checkpoint(load_path, **vars(args))
        args.ckpt_path = load_path

    # Initialize logger
    logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)

    # Load callbacks
    callbacks = load_callbacks(args)

    trainer = Trainer(callbacks=callbacks,
        logger=logger,
        max_epochs=args.max_epochs
    )
    trainer.fit(model, data_module)



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='google_dataset', type=str)
    parser.add_argument('--data_dir', default='/mnt/datalsj/dual_pixel/data/google', type=str)
    parser.add_argument('--model_name', default='dpnet', type=str)
    parser.add_argument('--loss', default='warp', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    
    # Model Hyperparameters
    parser.add_argument('--hid', default=64, type=int)
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--in_channel', default=3, type=int)
    parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # Training Control
    parser.add_argument('--max_epochs', default=100, type=int)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]

    main(args)
