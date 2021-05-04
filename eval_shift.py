import matplotlib.pyplot as plt
import numpy as np 

import sys
from data import * 
from module import * 
from pytorch_lightning.metrics.functional import accuracy

import pickle
from argparse import Namespace

def model_accuracy(model, data, labels):
	output = model(data)
	_, y_hat = torch.max(output, dim=1)
	test_acc = accuracy(y_hat.cpu(), labels.cpu())
	return test_acc



def eval_on_dataset_shift(dict_args):

	# load model state_dic
	args = Namespace(**dict_args)
	base_model = CIFAR10Module(args).model
	base_model.load_state_dict(torch.load(dict_args['model_path']))
	base_model = base_model.eval()

	data_module = CIFAR10Data(args)

	pre_activations = {}
	def get_pre_activation(layer_name):
		def hook(model, input):
			pre_activations[layer_name] = input.detach()
		return hook

	activations = {}
	def get_activation(layer_name):
		def hook(model, input, output):
			activation[layer_name] = output.detach()
		return hook

	#register hooks to look at activations
	layer = getattr(base_model, f"layer{dict_args['layer']}")
	max_layer_block = len(list(layer.children()))-1
	rbf_name = f"{max_layer_block}.RBF_activation"

	if dict_args['pre_activation']:
		base_model.register_forward_pre_hook(get_pre_activation(rbf_name))
	else:
		base_model.register_forward_hook(get_activation(rbf_name))

	model_acc_list = []
	output_list = []
	if dict_args['shift'] == "roll":

		for roll in np.arange(0,30,2):
			print(f"Processing roll {roll}...")
			data, labels = data_module.get_roll_data(int(roll))

			roll_acc = model_accuracy(base_model, data, labels)
			model_acc_list.append(roll_acc)

			base_model(data)
			model_outputs = pre_activations[rbf_name] if dict_args['pre_activation'] else activations[rbf_name]
			output_list.append(dict(roll_pix=roll, output=model_outputs))

	if dict_args['rot']:

		for rot in np.arange(0,360,15):
			print(f"Processing rot {rot}...")
			data, labels = data_module.get_rotation_data(rot)

			rot_acc = model_accuracy(base_model, data, labels)
			model_acc_list.append(rot_acc)
		
			base_model(data)
			model_outputs = pre_activations[rbf_name] if dict_args['pre_activation'] else activations[rbf_name]
			output_list.append(dict(roll_pix=roll, output=model_outputs))

	# model_name.pth
	pre, ext = os.path.splitext(dict_args['model_name'])
	model_dir = dict_args['model_dir']
	shift = dict_args['shift']
	act = 'pre_activation' if dict_args['pre_activation'] else 'post_activation'
	print(model_acc_list)
	with open(f"{model_dir}/{pre}_{shift}_acc.pkl", 'wb') as pickle_file:
		pickle.dump(model_acc_list, pickle_file)
	with open(f"{model_dir}/{pre}_{shift}_layer_{dict_args['layer']}_{act}_ouputs.pkl", 'wb') as pickle_file:
		pickle.dump(output_list, pickle_file)





def main(): 
	parser = argparse.ArgumentParser("Eval model on shifted data")

	parser.add_argument("--data_dir", default="data/cifar10/")
	parser.add_argument("--model_dir", default='/mnt/medqaresourcegroupdiag/medqa-fileshare/users/bk117/models')
	parser.add_argument("--model_name", required=True)
	parser.add_argument("--classifier", type=str, default="resnet18_RBF")
	parser.add_argument("--shift", type=str, choices=['roll', 'rot', 'cifar10_c'])
	parser.add_argument("--layer", type=int)
	parser.add_argument("--pre_activation", action="store_true")

	parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--max_epochs", type=int, default=100)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--gpu_id", type=str, default="0")

	parser.add_argument("--learning_rate", type=float, default=1e-2)
	parser.add_argument("--weight_decay", type=float, default=1e-2)

	args = parser.parse_args()
	dict_args = vars(args) 
	dict_args['model_path'] = os.path.join(dict_args['model_dir'], dict_args['model_name'])

	eval_on_dataset_shift(dict_args)

if __name__ == "__main__":
	main()