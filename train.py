"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import random
from collections import OrderedDict
from json import dumps

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

import util
from args import get_train_args
from models import init_training
from util import SQuAD


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, args.dataset, mode="train")
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get data loader
    log.info('Building dataset...')
    collate_fn = None if args.name in ['qanet'] else util.collate_fn
    train_loader, train_size, dev_loader, dev_size = build_datasets(args, collate_fn)
    dev_eval_dict = util.load_eval_file(args, args.dev_eval_file)

    # Get model
    log.info(f'Building {args.name} model...')
    config = None
    if args.model_config_file:
        with open(args.model_config_file, 'r') as pf: config = json_load(pf)
        log.info(f"Model config: {dumps(config, indent=4, sort_keys=True)}")
    model, optimizer, scheduler, ema, step = init_training(args, *(load_embeddings(args)), device, config=config)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // train_size
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), tqdm(total=train_size) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                # if use_char_vectors:
                #     log_p1, log_p2 = model(cw_idxs.to(device), qw_idxs.to(device))
                # else:
                log_p1, log_p2 = model(cw_idxs.to(device), cc_idxs.to(device), qw_idxs.to(device), qc_idxs.to(device))

                y1, y2 = y1.to(device), y2.to(device)
                nll_loss_1 = F.nll_loss(log_p1, y1)
                nll_loss_2 = F.nll_loss(log_p2, y2)
                loss = nll_loss_1 + nll_loss_2
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(epoch=epoch, STEP=util.millify(step), LR=current_lr, NLL=loss_val)

                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               current_lr,
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    dev_eval_file = util.preprocessed_path(args.dev_eval_file, args.data_dir, args.dataset)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)

                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_dict=dev_eval_dict,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def build_datasets(args, collate_fn=None):
    train_loader, train_size = build_loader(args.train_record_file, args.data_dir, args.dataset, args.batch_size, args.num_workers, True, args.use_squad_v2, collate_fn)
    dev_loader, dev_size = build_loader(args.dev_record_file, args.data_dir, args.dataset, args.batch_size, args.num_workers, False, args.use_squad_v2, collate_fn)
    return train_loader, train_size, dev_loader, dev_size


def build_loader(record_file, data_dir, dataset, batch_size, num_workers, shuffle, use_squad_v2=True, collate_fn=None):
    record_file_path = util.preprocessed_path(record_file, data_dir, dataset)
    dataset = SQuAD(record_file_path, use_squad_v2)
    num_samples = len(dataset)
    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader, num_samples


def load_embeddings(args):
    word_emb_file = util.preprocessed_path(args.word_emb_file, args.data_dir, args.dataset)
    char_emb_file = util.preprocessed_path(args.char_emb_file, args.data_dir, args.dataset)
    word_vectors = util.torch_from_json(word_emb_file)
    char_vectors = util.torch_from_json(char_emb_file)
    return word_vectors, char_vectors


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs.to(device), cc_idxs.to(device), qw_idxs.to(device), qc_idxs.to(device))
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
