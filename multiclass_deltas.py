import torch as ch
import utils
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm


def classwise_closed_form_solutions(logits, weights):
	# Iterate through all possible classes, calculate flip probabilities
	actual_label = ch.argmax(logits)
	delta_values = logits[actual_label] - logits
	delta_values /= weights - weights[actual_label]
	delta_values[actual_label] = np.inf
	return delta_values


if __name__ == "__main__":
	import sys

	model_arch = sys.argv[1]
	model_type = sys.argv[2]
	dataset    = sys.argv[3]
	filename   = sys.argv[4]

	if dataset == 'cifar10':
		dx = utils.CIFAR10()
		batch_size = 1024
	elif dataset == 'imagenet':
		batch_size = 256
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds = dx.get_dataset()
	model = dx.get_model(model_type, model_arch)

	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	weights_name = utils.get_logits_layer_name(model_arch)
	if not weights_name:
		raise ValueError("Architecture not supported yet")

	# Extract final weights matrix from model
	weights = None
	for name, param in model.state_dict().items():
		if name == weights_name:
			weights = param
			break

	n_features = weights.shape[1]
	sensitivities = {}
	# Get batches of data
	for (im, label) in test_loader:
		with ch.no_grad():
			(logits, features), _ = model(im.cuda(), with_latent=True)
		# For each data point in batch
		for j, logit in tqdm(enumerate(logits)):
			# For each feature
			for i in range(n_features):
				specific_weights = weights[:, i]
				# Get sensitivity values across classes
				sensitivity = classwise_closed_form_solutions(logit, specific_weights)

				# Only consider delta values that correspond to valud ReLU range, register others as 'inf'
				valid_sensitivity = sensitivity[features[j][i] + sensitivity >= 0]
				best_delta = ch.argmin(ch.abs(valid_sensitivity))
				best_sensitivity = valid_sensitivity[best_delta]
				best_sensitivity = best_sensitivity.cpu().numpy()
				sensitivities[i] = sensitivities.get(i, []) + [best_sensitivity]

	with open("%s.txt" % filename, 'w') as f:
		for i in range(n_features):
			floats_to_string = ",".join([str(x) for x in sensitivities[i]])
			f.write(floats_to_string + "\n")
