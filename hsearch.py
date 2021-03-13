import os

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

import train
import util

from helpers import GridSearch

def load_training_data(args):
    return util.SQuAD(util.preprocessed_path(args.train_record_file, args.data_dir, args.dataset), args.use_squad_v2)


def kfold_generator(args, splits, dataset):
    import torch.utils.data as data
    from sklearn.model_selection import KFold
    splitter = KFold(n_splits=splits, shuffle=True)
    for fold_index, (train_subset, test_subset) in enumerate(splitter.split(dataset)):
        train_sampler = data.SubsetRandomSampler(train_subset)
        test_sampler = data.SubsetRandomSampler(test_subset)

        train_loader = torch.utils.data.DataLoader(
            dataset, sampler=train_sampler,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=None
        )
        test_loader = torch.utils.data.DataLoader(
            dataset, sampler=test_sampler,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=None
        )

        yield fold_index, train_loader, len(train_subset), test_loader, len(test_subset)


def create_training_function(args, experiment_save_dir, k_fold_spits=None):
    device, args.gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(args.gpu_ids))
    word_vectors, char_vectors = train.load_embeddings(args)
    training_dataset = util.SQuAD(util.preprocessed_path(args.train_record_file, args.data_dir, args.dataset), args.use_squad_v2)
    eval_dataset = util.SQuAD(util.preprocessed_path(args.dev_record_file, args.data_dir, args.dataset), args.use_squad_v2)
    train_gold_dict = util.load_eval_file(args, args.train_eval_file)
    eval_gold_dict = util.load_eval_file(args, args.dev_eval_file)

    k_fold_spits = args.k_fold
    min_nll_decrease = args.min_nll_decrease

    def process_sample(sample, model, gold_dict=None):
        cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids = sample
        batch_size = cw_idxs.size(0)
        log_p1, log_p2 = model(cw_idxs.to(device), cc_idxs.to(device), qw_idxs.to(device), qc_idxs.to(device))
        y1, y2 = y1.to(device), y2.to(device)
        nll_loss_1 = F.nll_loss(log_p1, y1)
        nll_loss_2 = F.nll_loss(log_p2, y2)
        loss = nll_loss_1 + nll_loss_2
        preds = None
        if gold_dict:
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)
            preds, _ = util.convert_tokens(gold_dict, ids.tolist(), starts.tolist(), ends.tolist(), args.use_squad_v2)

        return loss, batch_size, preds

    def run_experiment(tbx, train_loader, train_size, eval_loader, eval_size, gold_dict, config):
        from models import init_training
        max_grad_norm = args.max_grad_norm
        model, optimizer, scheduler, ema, step = init_training(args, word_vectors, char_vectors, device, config)

        prev_epoch_avg_nll = None
        for epoch in range(step, args.num_epochs):
            model.train()
            epoch_avg_nll = util.AverageMeter()
            with torch.enable_grad(), tqdm(total=train_size) as progress_bar:
                for sample in train_loader:
                    loss, batch_size, _ = process_sample(sample, model, None)
                    nll = loss.item()
                    epoch_avg_nll.update(nll)
                    tbx.add_scalar('train/NLL', loss.item(), step)
                    current_lr = optimizer.param_groups[0]['lr']
                    tbx.add_scalar('train/LR', current_lr, step)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    ema(model, step // batch_size)
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch, STEP=util.millify(step), LR=current_lr, NLL=nll)
                    step += batch_size

            model.eval()
            ema.assign(model)
            results, pred_dict = evaluate(model, eval_loader, eval_size, gold_dict)
            ema.resume(model)

            tbx.add_scalar('eval/NLL', results['NLL'], step)
            if 'AvNA' in results:
                tbx.add_scalar('eval/AvNA', results['AvNA'], step)
            tbx.add_scalar('eval/F1', results['F1'], step)
            tbx.add_scalar('eval/EM', results['EM'], step)

            dev_eval_file = util.preprocessed_path(args.dev_eval_file, args.data_dir, args.dataset)
            util.visualize(tbx,
                           pred_dict=pred_dict,
                           eval_dict=gold_dict,
                           step=step,
                           split='eval',
                           num_visuals=args.num_visuals)

            if ((min_nll_decrease is not None) and (prev_epoch_avg_nll is not None) and (epoch_avg_nll.avg > prev_epoch_avg_nll - min_nll_decrease)):
                print(f"Avg NLL {epoch_avg_nll.avg:.2f} > {prev_epoch_avg_nll:.2f} - {(min_nll_decrease):.2f}. Break")
                break
            prev_epoch_avg_nll = epoch_avg_nll.avg

        return model, step

    def evaluate(model, eval_loader, eval_size, gold_dict):
        pred_dict = {}
        with torch.no_grad(), tqdm(total=eval_size) as progress_bar:
            nll_meter = util.AverageMeter()
            for sample in eval_loader:
                loss, batch_size, preds = process_sample(sample, model, gold_dict)
                nll_meter.update(loss.item(), batch_size)
                pred_dict.update(preds)
                progress_bar.update(batch_size)
                progress_bar.set_postfix(NLL=nll_meter.avg)

            results = {
                **util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2),
                **{'NLL': nll_meter.avg}
            }
        return results, pred_dict



    def kfold_training_function(experiment, config):
        avg_meter = util.MultiAverageMeter(['F1', 'EM', 'AvNA', 'NLL'])
        gold_dict = train_gold_dict
        for fold_index, train_loader, train_size, test_loader, test_size in kfold_generator(args, k_fold_spits, training_dataset):
            save_dir = os.path.join(experiment_save_dir, *GridSearch.experiment_path(experiment), f"fold={fold_index + 1}")
            tbx = SummaryWriter(save_dir)

            model, steps = run_experiment(tbx, train_loader, train_size, test_loader, test_size, gold_dict, config)
            results, _ = evaluate(model, test_loader, test_size, gold_dict)
            avg_meter.update(results, steps)

        return {
            **experiment,
            **avg_meter.avg
        }

    def training_function(experiment, config):
        import torch.utils.data as data
        train_loader = data.DataLoader(training_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=None)
        eval_loader = data.DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=None)
        save_dir = os.path.join(experiment_save_dir, *GridSearch.experiment_path(experiment))
        tbx = SummaryWriter(save_dir)

        train_size = len(training_dataset)
        eval_size = len(eval_dataset)
        model, steps = run_experiment(tbx, train_loader, train_size, eval_loader, eval_size, eval_gold_dict, config)
        results, _ = evaluate(model, eval_loader, eval_size, eval_gold_dict)

        return {
            **experiment,
            **results
        }

    return kfold_training_function if k_fold_spits is not None else training_function

def main(args):
    import pandas as pd

    experiment_save_dir = util.get_save_dir(
        args.save_dir, args.name, args.dataset, mode="hyper"
    )
    training_function = create_training_function(args, experiment_save_dir)

    grid_search = GridSearch.load(args.experiments_file)
    results = [
        training_function(experiment, config)
        for experiment, config in grid_search.random_experiments(args.max_experiments)
    ]

    df = pd.DataFrame(results)
    df.to_csv("hyperparams.csv")

    print("Best config: ", df.sort_values('F1', ascending=False).iloc[0])


if __name__ == '__main__':
    from args import get_hsearch_args

    main(get_hsearch_args())


