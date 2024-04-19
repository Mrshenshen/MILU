import re
import math
from argparse import ArgumentParser, Namespace
from hydra.utils import instantiate
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.plugins import DDPPlugin
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from model.ours.dataset import JointDataModule

from pytorch_lightning import Callback

from torch.optim import Adam
from train_helper import newLightningModule

def dict_parser(s: str):
    return eval('{' + re.sub(r'(\w+)=(["\']?\w+["\']?)', r'"\1":\2', s) + '}')

def add_common_trainer_util_args(parser, default_monitor_variable='val_loss', default_monitor_mode='min'):
    if default_monitor_mode not in ['min', 'max']:
        raise ValueError(default_monitor_mode)
    parser.add_argument('--lr_find_kwargs', default=dict(min_lr=5e-6, max_lr=1e-2), type=dict_parser,
                        help='Arguments for LR find (--auto_lr_find). Default "min_lr=5e-6,max_lr=1e-2"')
    parser.add_argument('--random_seed', default=42, type=lambda s: None if s == 'None' else int(s),
                        help='Seed everything. Set to "None" to disable global seeding')
    parser.add_argument('--auto_resume', default=False, action='store_true',
                        help='Automatically resume last saved checkpoint, if available.')
    parser.add_argument('--test_only', default=False, action='store_true',
                        help='Skip fit and call only test. This implies automatically detecting newest checkpoint, '
                             'if --checkpoint_path is not given.')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='Load this checkpoint to resume training or run testing. '
                             'Pass in the special value "best" to use the best checkpoint according to '
                             'args.monitor_variable and args.monitor_mode. '
                             'Using "best" only works with test_only mode.')
    parser.add_argument('--ignore_existing_checkpoints', default=False, action='store_true',
                        help='Proceed even with training a new model, even if previous checkpoints exists.')
    parser.add_argument('--monitor_variable', default=default_monitor_variable, type=str,
                        help='Variable to monitor for early stopping and for checkpoint selection. '
                             f'Default: {default_monitor_variable}')
    parser.add_argument('--monitor_mode', default=default_monitor_mode, type=str, choices=['min', 'max'],
                        help='Mode for monitoring the monitor_variable (for early stopping and checkpoint selection). '
                             f'Default: {default_monitor_mode}')
    parser.add_argument('--reset_early_stopping_criterion', default=False, action='store_true',
                        help='Reset the early stopping criterion when loading from checkpoint. '
                             'Prevents immediate exit after switching to more complex dataset in curriculum strategy')

def apply_argparse_defaults_to_hydra_config(config: DictConfig, parser: ArgumentParser, verbose=False):
    args = parser.parse_args([])  # Parser is not allowed to have required args, otherwise this will fail!
    defaults = vars(args)

    def _apply_defaults(dest: DictConfig, source: dict, indentation=''):
        for k, v in source.items():
            if k in dest and isinstance(v, dict):
                current_value = dest[k]
                if current_value is not None:
                    assert isinstance(current_value, DictConfig)
                    _apply_defaults(current_value, v, indentation + ' ')
            elif k not in dest:
                dest[k] = v
                if verbose:
                    print(indentation, 'set default value for', k)

    with open_dict(config):
        _apply_defaults(config, defaults)


def _adjust_ddp_config(trainer_cfg):
    trainer_cfg = dict(trainer_cfg)
    strategy = trainer_cfg.get('strategy', None)
    if trainer_cfg['gpus'] > 1 and strategy is None:
        strategy = 'ddp'  # Select ddp by default
    if strategy == 'ddp':
        trainer_cfg['strategy'] = DDPPlugin(
            find_unused_parameters=trainer_cfg['find_unused_parameters'], 
            gradient_as_bucket_view=True)
    return trainer_cfg


@hydra.main(config_path='config', config_name='base')
def train(config: DictConfig):
    fake_parser = ArgumentParser()
    add_common_trainer_util_args(fake_parser, default_monitor_variable='val_loss')
    apply_argparse_defaults_to_hydra_config(config.trainer, fake_parser)
    pl.seed_everything(config.trainer.random_seed, workers=True)
    trainer_cfg = Namespace(**_adjust_ddp_config(config.trainer))

    data = JointDataModule(config.dataset)
    data.setup()

    total_steps = trainer_cfg.max_epochs * math.floor(len(data.train_dataset) / trainer_cfg.gpus / config.dataset.batch_size)
    model = instantiate(config.model, max_v_len=config.dataset.max_v_len)
    if trainer_cfg.checkpoint_path:
        state_dict = torch.load(trainer_cfg.checkpoint_path, map_location='cpu')['state_dict']
        if not trainer_cfg.load_nlq_head:
            print('Train NLQ head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not "nlq_head" in k}
        if not trainer_cfg.load_decoder:
            print('Train LM decoder head from scratch')
            state_dict = {k: v for k, v in state_dict.items() if not ("decoder" in k or "lm_head" in k)}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f'Load checkpoint: {trainer_cfg.checkpoint_path}')
        print(f'Missing Keys: {missing_keys}')
        print(f'Unexpected Keys: {unexpected_keys}')

    dirpath = './ckpts/experiment'
    if trainer_cfg.test_only:  # evaluation
        trainer = pl.Trainer.from_argparse_args(
            trainer_cfg, 
            enable_checkpointing=False, 
            logger=False
        )
        if trainer_cfg.val:
            trainer.validate(
                model, data.val_dataloader(),
            )
        else:
            trainer.test(
                model, data.test_dataloader(),
            )
    else:  # training
        print(len(data.train_dataloader()))
        plm = newLightningModule(config, total_steps)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = Adam(params=model.parameters(), lr=0.00005, weight_decay=0.0)
        lr_scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=0.5)
        for epoch in range(trainer_cfg.max_epochs):
            epoch_loss = 0
            best_rogue_epoch = [0,0]
            best_closs_epoch = [0,0]
            best_R1_03_epoch = [0,0]
            for index, data in enumerate(data.train_dataloader()[:20]):
                # data = data.to(device)
                optimizer.zero_grad()
                loss, _, _ = model(**data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
                lr_scheduler.step()
            epoch_loss /= len(data.train_dataloader())
            print(f"current epoch: {epoch}, current loss: {epoch_loss}")
            for index, data in enumerate(data.val_dataloader()):
                nlq_results, answer_tokens = model.generate(**data)
                val_ret = {
                    'question': data['q_text'],
                    'video_id': data['video_id'],
                    'answer': data['a_text'] if 'a_text' in data else '',
                    'nlq_results': nlq_results,
                    'query_id': data['query_id'],
                    'sample_ratio': data['sample_ratio'],
                    'task': data['task']
                }
                res = plm.validation_epoch_end(val_ret)
                if best_closs_epoch[0] < res['val_close_acc']:
                    best_closs_epoch[0] = res['val_close_acc']
                    best_closs_epoch[1] = epoch
                if best_rogue_epoch[0] < res['val_ROUGE']:
                    best_rogue_epoch[0] = res['val_ROUGE']
                    best_rogue_epoch[1] = epoch
                if best_R1_03_epoch[0] < res['val_R1_03']:
                    best_R1_03_epoch[0] = res['val_R1_03']
                    best_R1_03_epoch[1] = epoch
                    # ['val_R1_03'] ['val_ROUGE']
                print(f'best_rogue_epoch: {best_rogue_epoch}')
                print(f'best_closs_epoch: {best_closs_epoch}')
                print(f'best_R1_03_epoch: {best_R1_03_epoch}')
                torch.save(model.state_dict(), f'./ckpts/experiment/epoch_{epoch}.pth')


    
if __name__ == '__main__':
    train()
