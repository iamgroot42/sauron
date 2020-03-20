import torch as ch
import numpy as np
from robustness.model_utils import make_and_restore_model
from robustness.tools.helpers import save_checkpoint
import sys

import utils


def chuck_inf_means(senses):
	chucked = []
	for i in range(senses.shape[0]):
		x = senses[i]
		chucked.append(np.mean(x[x != np.inf]))
	return np.array(chucked)


if __name__ == "__main__":

	model_arch  = sys.argv[1]
	model_type  = sys.argv[2]
	random_drop = sys.argv[3] == 'random'
	num_drop    = int(sys.argv[4])
	model_path  = sys.argv[5]

	if random_drop not in ['random', 'most', 'least']:
		raise ValueError("Method of selecting neurons to drop not supported")

	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	model_kwargs = {
		'arch': model_arch,
		'dataset': ds,
		'resume_path': model_type
	}

	# Get scaled delta values
	senses = constants.get_deltas(model_type, model_arch)
	(mean, std) = constants.get_stats(model_type, model_arch)

	# Load model
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()
	
	print("Dropping %d out of %d neurons" % (num_drop, senses.shape[0]))

	# Random weight drop-out if negative factor
	if random_drop:
		print("Random drop-out!")
		worst_n = np.random.permutation(senses.shape[0])[:num_drop]
	else:
		# 99.7% interval
		threshold = mean + 3 * std

		# Only consider neurons with any hopes of attackking (delta within some sensible range)
		senses = utils.scaled_values(senses, mean, std)
		senses = chuck_inf_means(senses)

		if random_drop == 'most':
			worst_n = np.argsort(np.abs(senses))[:num_drop]
		else:
			worst_n = np.argsort(-np.abs(senses))[:num_drop]

	# Extract final weights matrix from model
	with ch.no_grad():
		model.state_dict().get("module.model.classifier.weight")[:, worst_n] = 0

	# Save modified model
	sd_info = {
		'model': model.state_dict(),
		'epoch': 1
	}
	save_checkpoint(sd_info, False, model_path)
