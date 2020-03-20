import os
import torch as ch
from robustness.model_utils import make_and_restore_model
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import utils


def pgd_optimization(model, inp_og, target_rep, indices_mask, eps, random_restart_targets, iters=100,
	reg_weight=1e0, p='2', verbose=True, custom_best=False, fake_relu=True, random_restarts=0):
	# Modified inversion loss that puts emphasis on non-matching neurons to have similar activations
	def custom_inversion_loss(m, inp, targ):
		output, rep = m(inp, with_latent=True, fake_relu=fake_relu)
		# Normalized L2 error w.r.t. the target representation
		loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
		# Extra loss term (normalized)
		aux_loss = ch.sum(ch.abs((rep - targ) * indices_mask), dim=1)
		aux_loss = ch.div(aux_loss, ch.norm(targ * indices_mask, dim=1))
		# Lagrangian formulation:
		return loss + reg_weight * aux_loss, output

	if custom_best:
		# If True, use the 'only neuron i' based 'best' evaluation
		if custom_best is True:
			def custom_loss_fn(loss, x):
				# Check how much beyond minimum delta the  perturbation on i^th index is
				# Negative sign, since we want higher delta-diff to score better
				(_, rep), _ = model(x, with_latent=True, fake_relu=fake_relu)
				return - ch.sum((rep - target_rep) * indices_mask, dim=1)
			custom_best = custom_loss_fn
		# Else, expect custom_best function to be passed along
	else:
		# If nothing passed along, use simple comparison
		custom_best = None


	kwargs = {
		'custom_loss': custom_inversion_loss,
		'constraint': p,
		'eps': eps,
		'step_size': 2.5 * eps / iters,
		'iterations': iters,
		'targeted': True,
		'do_tqdm': verbose,
		'custom_best': custom_best,
		'random_restarts': random_restarts,
		'random_restart_targets': random_restart_targets
	}
	_, im_matched = model(inp_og, target_rep, make_adv=True, **kwargs)
	return im_matched


def find_impostors(model, delta_values, ds, images, mean, std,
	verbose=True, n=4, eps=2.0, iters=200,
	norm='2', custom_best=False, fake_relu=True,
	analysis_start=0, random_restarts=0, delta_analysis=False):
	image_ = []
	# Get target images
	for image in images:
		targ_img = image.unsqueeze(0)
		real = targ_img.repeat(n, 1, 1, 1)
		image_.append(real)
	real = ch.cat(image_, 0)

	# Get scaled senses
	scaled_delta_values = utils.scaled_values(delta_values, mean, std, eps=0)
	# Replace inf values with largest non-inf values
	delta_values[delta_values == np.inf] = delta_values[delta_values != np.inf].max()

	# Pick easiest-to-attack neurons per image
	easiest = np.argsort(scaled_delta_values, axis=0)

	# Get feature representation of current image
	with ch.no_grad():
		(_, image_rep), _  = model(real.cuda(), with_latent=True)

	# Construct delta vector and indices mask
	delta_vec = ch.zeros_like(image_rep)
	indices_mask = ch.zeros_like(image_rep)
	for j in range(len(images)):
		for i, x in enumerate(easiest[analysis_start : analysis_start + n, j]):
			delta_vec[i + j * n, x] = delta_values[x, j]
			indices_mask[i + j * n, x] = 1		

	impostors = parallel_impostor(model, delta_vec, real, indices_mask, verbose,
		eps, iters, norm, custom_best, fake_relu, random_restarts)

	with ch.no_grad():
		if delta_analysis:
			(pred, latent), _ = model(impostors, with_latent=True)
		else:
			pred, _ = model(impostors)
			latent = None
	label_pred = ch.argmax(pred, dim=1)

	clean_pred, _ = model(real)
	clean_pred = ch.argmax(clean_pred, dim=1)

	clean_preds = clean_pred.cpu().numpy()
	preds       = label_pred.cpu().numpy()

	succeeded = [[] for _ in range(len(images))]
	if delta_analysis:
		delta_succeeded = [[] for _ in range(len(images))]
	for i in range(len(images)):
		for j in range(n):
			succeeded[i].append(preds[i * n + j] != clean_preds[i * n + j])
			if delta_analysis:
				analysis_index = easiest[analysis_start : analysis_start + n, i][j]
				success_criterion = (latent[i * n + j] >= (image_rep[i * n + j] + delta_vec[i * n + j]))
				delta_succeeded[i].append(success_criterion[analysis_index].cpu().item())
	succeeded = np.array(succeeded)
	if delta_analysis:
		delta_succeeded = np.array(delta_succeeded, 'float')
	image_labels = [clean_preds, preds]

	if not delta_analysis:
		delta_succeeded = None

	return (real, impostors, image_labels, succeeded, None, delta_succeeded)


def parallel_impostor(model, delta_vec, im, indices_mask, verbose, eps,
	iters, norm, custom_best, fake_relu, random_restarts):
	# Get feature representation of current image
	with ch.no_grad():
		(target_logits, image_rep), _  = model(im.cuda(), with_latent=True, fake_relu=fake_relu)
		target_logits = ch.argmax(target_logits, dim=1)

	# Get target feature rep
	target_rep = image_rep + delta_vec

	# Override custom_best, use cross-entropy on model instead
	criterion = ch.nn.CrossEntropyLoss(reduction='none').cuda()
	def ce_loss(loss, x):
		output, _ = model(x, fake_relu=fake_relu)
		# We want CE loss b/w new and old to be as high as possible
		return -criterion(output, target_logits)
	# Use CE loss
	if custom_best: custom_best = ce_loss

	im_matched = pgd_optimization(model, im, target_rep, indices_mask,
		random_restart_targets=target_logits, eps=eps, iters=iters, verbose=verbose,
		p=norm, reg_weight=1e1, custom_best=custom_best, fake_relu=fake_relu,
		random_restarts=random_restarts)
	
	return im_matched


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='nat', help='type of model (nat/l2/linf)')
	parser.add_argument('--eps', type=float, default=0.5, help='epsilon-iter')
	parser.add_argument('--iters', type=int, default=50, help='number of iterations')
	parser.add_argument('--n', type=int, default=16, help='number of neurons per image')
	parser.add_argument('--bs', type=int, default=4, help='batch size while performing attack')
	parser.add_argument('--custom_best', type=bool, default=True, help='look at absoltue loss or perturbation for best-loss criteria')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [cifar10, imagenet]')
	parser.add_argument('--norm', type=str, default='2', help='P-norm to limit budget of adversary')
	parser.add_argument('--analysis', type=bool, default=False, help='report neuron-wise attack success rates?')
	parser.add_argument('--delta_analysis', type=bool, default=False, help='report neuron-wise delta-achieve rates?')
	parser.add_argument('--random_restarts', type=int, default=0, help='how many random restarts? (0 -> False)')
	parser.add_argument('--analysis_start', type=int, default=0, help='index to start from (to capture n). used only when analysis flag is set')
	
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	batch_size      = args.bs
	iters           = args.iters
	eps             = args.eps
	n               = args.n
	norm            = args.norm
	custom_best     = args.custom_best
	fake_relu       = (model_arch != 'vgg19')
	analysis        = args.analysis
	delta_analysis  = args.delta_analysis
	analysis_start  = args.analysis_start
	random_restarts = args.random_restarts

	# Load model
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
	else:
		print("Invalid Dataset Specified")
	ds = constants.get_dataset()

	# Load model
	model = constants.get_model(model_type , model_arch)
	# Get stats for neuron activations
	senses = constants.get_deltas(model_type, model_arch)
	(mean, std) = constants.get_stats(model_type, model_arch)

	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	index_base, avg_successes = 0, 0
	attack_rates = [0, 0, 0, 0]
	impostors_latents = []
	all_impostors = []
	neuron_wise_success = []
	delta_wise_success  = []
	iterator = tqdm(test_loader)
	for (image, _) in iterator:
		picked_indices = list(range(index_base, index_base + len(image)))
		(real, impostors, image_labels, succeeded, impostors_latent, delta_succeeded) = find_impostors(model, senses[:, picked_indices], ds,
															image.cpu(), mean, std, n=n, verbose=False,
															eps=eps, iters=iters, norm=norm,
															custom_best=custom_best, fake_relu=fake_relu,
															analysis_start=analysis_start, random_restarts=random_restarts,
															delta_analysis=delta_analysis)

		attack_rates[0] += np.sum(np.sum(succeeded[:, :1], axis=1) > 0)
		attack_rates[1] += np.sum(np.sum(succeeded[:, :4], axis=1) > 0)
		attack_rates[2] += np.sum(np.sum(succeeded[:, :8], axis=1) > 0)
		num_flips = np.sum(succeeded, axis=1)
		attack_rates[3] += np.sum(num_flips > 0)
		avg_successes += np.sum(num_flips)
		index_base += len(image)
		# Keep track of attack success rate
		iterator.set_description('(n=1,4,8,%d) Success rates : (%.2f, %.2f, %.2f, %.2f) | | Flips/Image : %.2f/%d' \
			% (n, 100 * attack_rates[0]/index_base,
				100 * attack_rates[1]/index_base,
				100 * attack_rates[2]/index_base,
				100 * attack_rates[3]/index_base,
				avg_successes / index_base, n))
		# Keep track of neuron-wise attack success rate
		if analysis:
			neuron_wise_success.append(succeeded)
		if delta_analysis:
			delta_wise_success.append(delta_succeeded)

	if analysis:
		neuron_wise_success = np.concatenate(neuron_wise_success, 0)
		neuron_wise_success = np.mean(neuron_wise_success, 0)
		for i in range(neuron_wise_success.shape[0]):
			print("Neuron %d attack success rate : %f %%" % (i + analysis_start, 100 * neuron_wise_success[i]))
		print()

	if delta_analysis:
		delta_wise_success = np.concatenate(delta_wise_success, 0)
		delta_wise_success = np.mean(delta_wise_success, 0)
		for i in range(delta_wise_success.shape[0]):
			print("Neuron %d acheiving-delta success rate : %f %%" % (i + analysis_start, 100 * delta_wise_success[i]))
		print()

	print("Attack success rate : %f %%" % (100 * attack_rates[-1]/index_base))
	print("Average flips per image : %f/%d" % (avg_successes / index_base, n))
