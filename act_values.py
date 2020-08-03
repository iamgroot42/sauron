import torch as ch
import utils
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
	import sys

	model_arch   = sys.argv[1]
	model_type   = sys.argv[2]
	prefix       = sys.argv[3]
	dataset      = sys.argv[4]

	if dataset == 'cifar10':
		dx = utils.CIFAR10()
	elif dataset == 'imagenet':
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds = dx.get_dataset()
	model = dx.get_model(model_type, model_arch)

	batch_size = 128
	all_reps = []
	train_loader = None
	if dataset == 'cifar10':
		train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)
	else:
		_, val_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True)

	def get_reps(data_loader):
		for (im, label) in tqdm(data_loader):
			with ch.no_grad():
				(_, rep), _ = model(im.cuda(), with_latent=True)
				all_reps.append(rep.cpu())

	if train_loader:
		get_reps(train_loader)
	get_reps(val_loader)

	all_reps = ch.cat(all_reps)
	ch_mean  = ch.mean(all_reps, dim=0)
	ch_std   = ch.std(all_reps, dim=0)

	# Dump mean, std vectors for later use:
	np_mean = ch_mean.cpu().numpy()
	np_std  = ch_std.cpu().numpy()
	np.save(prefix + "feature_mean", np_mean)
	np.save(prefix + "feature_std",   np_std)
