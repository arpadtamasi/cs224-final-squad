import math
from dataclasses import dataclass

import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch import nn

import util
from helpers import load_dataclass


def init_training(args, word_vectors, char_vectors, device, config=None):
    config = config or {}
    model_name = args.name
    if model_name == 'baseline':
        model, optimizer, scheduler = init_baseline_training(args, word_vectors, config)
    else:
        if model_name == 'claf':
            model, optimizer, scheduler = init_claf_training(args, char_vectors, word_vectors, config)
        elif model_name == 'qanet':
            model, optimizer, scheduler = init_qanet_training(args, char_vectors, word_vectors, config)
        elif model_name == 'qanet2':
            model, optimizer, scheduler = init_qanet2_training(args, char_vectors, word_vectors, config)
        elif model_name == 'qanet2-performer':
            model, optimizer, scheduler = init_qanet2_training(args, char_vectors, word_vectors, config, use_performer=True)
        else:
            raise Exception("Unknown model")

    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()

    ema_decay = args.ema_decay
    ema = util.EMA(model, ema_decay)

    return model, optimizer, scheduler, ema, step


def init_baseline_training(args, word_vectors, config):
    from models.bidaf import BiDAF

    model_config = config.get('model', {})
    bidaf_conf: BiDAFConf = load_dataclass(model_config, BiDAFConf)

    model = BiDAF(
        word_vectors=word_vectors, hidden_size=bidaf_conf.hidden_size, drop_prob=bidaf_conf.drop_prob
    )

    optimizer_config = config.get('optimizer', {})
    adadelta_config = load_dataclass(optimizer_config, AdadeltaConf)
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=adadelta_config.learning_rate,
        weight_decay=adadelta_config.weight_decay
    )

    scheduler_config = config.get('scheduler', {})
    multiplicative_scheduler_config: MultiplicativeSchedulerConf = load_dataclass(scheduler_config, MultiplicativeSchedulerConf)
    scheduler = sched.LambdaLR(optimizer, lambda batch: multiplicative_scheduler_config.multiplier ** ((batch * args.batch_size) // 1000))

    return model, optimizer, scheduler


def init_qanet_training(args, char_vectors, word_vectors, config):
    from models.qanet import QANet, QANetConf

    model_config = config.get('model', {})
    model = QANet(
        word_vectors=word_vectors,
        char_vectors=char_vectors,
        config=load_dataclass(model_config, QANetConf)
    )
    optimizer = __qanet_adam_optimizer(model, config)
    scheduler = __qanet_adam_scheduler(optimizer, config, args)
    return model, optimizer, scheduler

def init_qanet2_training(args, char_vectors, word_vectors, config, use_performer=False):
    from models.qanet2 import QANet, QANetConf

    model_config = config.get('model', {})
    model = QANet(
        word_mat=word_vectors,
        char_mat=char_vectors,
        config=load_dataclass(model_config, QANetConf),
        use_performer=use_performer
    )
    optimizer = __qanet_adam_optimizer(model, config)
    scheduler = __qanet_adam_scheduler(optimizer, config, args)
    return model, optimizer, scheduler

def init_claf_training(args, char_vectors, word_vectors, config):
    from models.claf import QANet, QANetConf

    model_config = config.get('model', {})
    model = QANet(
        word_vectors=word_vectors,
        char_vectors=char_vectors,
        config=load_dataclass(model_config, QANetConf)
    )
    optimizer = __qanet_adam_optimizer(model, config)
    scheduler = __qanet_adam_scheduler(optimizer, config, args)

    return model, optimizer, scheduler


def __qanet_adam_scheduler(optimizer, config, args):
    scheduler_config = config.get('scheduler', {})
    warmup_conf = load_dataclass(scheduler_config, WarmupSchedulerConf)
    scheduler = sched.LambdaLR(
        optimizer,
        lambda batch: min(
            1,
            1 / math.log(warmup_conf.warmup_steps - 1) * math.log(batch * args.batch_size + 1)
        )
    )
    return scheduler


def __qanet_adam_optimizer(model, config):
    optimizer_config = config.get('optimizer', {})
    adam_config = load_dataclass(optimizer_config, AdamConf)
    optimizer = optim.Adam(
        model.parameters(),
        lr=adam_config.learning_rate,
        betas=(adam_config.beta_1, adam_config.beta_2),
        eps=adam_config.eps,
        weight_decay=adam_config.weight_decay
    )
    return optimizer


@dataclass
class AdamConf:
    beta_1: float = 0.8
    beta_2: float = 0.999
    weight_decay: float = 3e-7
    eps: float = 1e-7
    learning_rate: float = 0.001


@dataclass
class WarmupSchedulerConf:
    warmup_steps: int = 1000

@dataclass
class BiDAFConf:
    hidden_size: int = 100
    drop_prob: float = 0.

@dataclass
class AdadeltaConf:
    learning_rate: float = 0.2
    weight_decay: float = 0.

@dataclass
class AdadeltaConf:
    learning_rate: float = 0.2
    weight_decay: float = 0.

@dataclass
class MultiplicativeSchedulerConf:
    multiplier: float = 1.
