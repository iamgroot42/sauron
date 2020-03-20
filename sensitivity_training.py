import torch as ch
import numpy as np
from robustness.train import train_model
from robustness.tools import helpers
from robustness import defaults
from robustness.defaults import check_and_fill_args
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS
import os
from itertools import combinations 
import cox
import utils
import argparse


def regularization_term(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal classification loss
	loss = train_criterion(logits, targets)
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		topk_diff, _ = ch.topk(ch.abs(diff[0] - diff[1]), top_k)
		diffs.append(ch.mean(topk_diff))
	first_term = ch.max(ch.stack(diffs, dim=0))

	diffs_2 = []
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = ch.abs(features * (diff_2_1 - diff_2_2) / features_norm)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs_2.append(use_these)
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)

	return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--top_k', type=int, default=16, help='top-k (neurons) considered while calculating loss terms')
	parser.add_argument('--start_lr', type=float, default=1e-2, help='starting LR for optimizer')
	parser.add_argument('--delta_1', type=float, default=1e1, help='loss coefficient for first term')
	parser.add_argument('--delta_2', type=float, default=1e2, help='loss coefficient for second term')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
	parser.add_argument('--output_dir', type=str, default='', help='path where model is to be saved')

	parsed_args = parser.parse_args()
	for arg in vars(parsed_args):
		print(arg, " : ", getattr(parsed_args, arg))

	def regularizer(model, inp, targets, train_criterion, adv, attack_kwargs):
		return regularization_term(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
			parsed_args.delta_2, train_criterion, adv, attack_kwargs)

	if not os.path.exists(parsed_args.output_dir):
		raise ValueError("Please provide valid save dir for model")

	train_kwargs = {
	    'out_dir': parsed_args.output_dir,
	    'adv_train': 0,
	    'exp_name': 'sensitivity_training',
	    'dataset': 'cifar',
    	'arch': 'vgg19',
    	'adv_eval': True,
    	'batch_size': parsed_args.batch_size,
    	# Validation-evaluation using PGD-L2 attack (to track L2 PGD perturbation robustness)
    	'attack_lr': (2.5 * 0.5) / 10,
    	'constraint': '2',
    	'eps': 0.5,
	    'attack_steps': 20,
    	'use_best': True,
    	'eps_fadein_epochs': 0,
    	'random_restarts': 0,
    	'lr': parsed_args.start_lr,
    	'use_adv_eval_criteria': 1,
    	'regularizer': regularizer,
    	'let_reg_handle_loss': True
	}

	ds_class = DATASETS[train_kwargs['dataset']]

	train_args = cox.utils.Parameters(train_kwargs)

	dx = utils.CIFAR10()
	dataset = dx.get_dataset()

	args = check_and_fill_args(train_args, defaults.TRAINING_ARGS, ds_class)
	args = check_and_fill_args(train_args, defaults.MODEL_LOADER_ARGS, ds_class)

	model, _ = make_and_restore_model(arch='vgg19', dataset=dataset)
	
	# Make the data loaders
	train_loader, val_loader = dataset.make_loaders(args.workers, args.batch_size, data_aug=bool(args.data_aug))

	# Prefetches data to improve performance
	train_loader = helpers.DataPrefetcher(train_loader)
	val_loader = helpers.DataPrefetcher(val_loader)

	store = cox.store.Store(args.out_dir, args.exp_name)
	args_dict = args.as_dict() if isinstance(args, cox.utils.Parameters) else vars(args)
	schema = cox.store.schema_from_dict(args_dict)
	store.add_table('metadata', schema)
	store['metadata'].append_row(args_dict)

	model = train_model(args, model, (train_loader, val_loader), store=store)
