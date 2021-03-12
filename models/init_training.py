import math

import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch import nn

import util


def init_training(args, word_vectors, char_vectors, device, config = None):
    config = config or {}
    model_name = args.name
    if model_name == 'baseline':
        model, optimizer, scheduler = init_baseline_training(args, word_vectors, config)
    else:
        if model_name == 'claf':
            model, optimizer, scheduler = init_claf_training(args, char_vectors, word_vectors, config)
        elif model_name == 'qanet':
            model, optimizer, scheduler = init_qanet_training(args, char_vectors, word_vectors, config)
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
    model = BiDAF(
        word_vectors=word_vectors, hidden_size=args.hidden_size, drop_prob=args.drop_prob
    )

    optimizer = optim.Adadelta(model.parameters(), args.lr, weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda batch: args.lr_decay ** ((batch * args.batch_size) // 1000))

    return model, optimizer, scheduler


def init_claf_training(args, char_vectors, word_vectors, config):
    from models.claf import QANet

    drop_prob = config.get("drop_prob", 0.1)
    layer_dropout = config.get("layer_dropout", 0.9)
    dropout_char = config.get("dropout_char", 0.05)

    model = QANet(
        word_vectors=word_vectors, char_vectors=char_vectors,
        aligned_query_embedding=True, freeze_char_embedding=False,

        model_dim=128,

        kernel_size_in_embedding=7,
        num_head_in_embedding=8,
        num_conv_block_in_embedding=4,
        num_embedding_encoder_block=1,

        kernel_size_in_modeling=5,
        num_head_in_modeling=8,
        num_conv_block_in_modeling=2,
        num_modeling_encoder_block=7,

        dropout=drop_prob, model_encoder_dropout=drop_prob,
        layer_dropout=layer_dropout,
        char_dropout=dropout_char
    )

    beta_1 = config.get("beta_1", 0.8)
    beta_2 = config.get("beta_2", 0.999)
    w2d = config.get("w2d", 3e-7)
    learning_rate = config.get("learning_rate", 0.0005)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta_1, beta_2),
        eps=10e-7,
        weight_decay=w2d
    )

    warmup_steps = config.get("warmup_steps", 1000)
    scheduler = sched.LambdaLR(
        optimizer,
        lambda batch: min(
            1,
            1 / math.log(warmup_steps - 1) * math.log(batch * args.batch_size + 1)
        )
    )

    return model, optimizer, scheduler


def init_qanet_training(args, char_vectors, word_vectors, config):
    from models.qanet import QANet
    drop_prob = config.get("drop_prob", 0.1)
    dropout_char = config.get("dropout_char", 0.05)
    d_model = config.get("d_model", 128)
    n_head = config.get("n_head", 8)

    len_c = args.para_limit
    len_q = args.ques_limit
    model = QANet(
        word_vectors=word_vectors, char_vectors=char_vectors,
        d_model=d_model, n_head=n_head,
        len_c=len_c, len_q=len_q,
        dropout=drop_prob, dropout_char=dropout_char, freeze_char_embedding=False
    )

    beta_1 = config.get("beta_1", 0.8)
    beta_2 = config.get("beta_2", 0.999)
    w2d = config.get("w2d", 3e-7)
    learning_rate = config.get("learning_rate", 0.0005)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta_1, beta_2),
        eps=10e-7,
        weight_decay=w2d
    )

    warmup_steps = config.get("warmup_steps", 1000)
    scheduler = sched.LambdaLR(
        optimizer,
        lambda batch: min(
            1,
            1 / math.log(warmup_steps - 1) * math.log(batch * args.batch_size + 1)
        )
    )

    return model, optimizer, scheduler
