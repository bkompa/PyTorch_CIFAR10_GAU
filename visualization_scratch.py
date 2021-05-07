import numpy as np 
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd

model_dir = '/mnt/medqaresourcegroupdiag/medqa-fileshare/users/bk117/models'
model_name = 'cifar10_val_4'

model_path = f"{model_dir}/{model_name}"
layer = 4
split = 'val'
activation = 'post'

val_activations = np.load(f"{model_path}/{model_name}_{split}_layer_{layer}_{activation}_activation_ouputs.npy")

def mean_log_activation(activations):
	return(np.mean(np.log(activations), axis=(-1,-2)))

mean_log_val_activations = mean_log_activation(val_activations)

sum_mean_log_val_activations = np.sum(mean_log_val_activations, axis=1)
val_df = pd.DataFrame(data=sum_mean_log_val_activations, columns=["log_sum"])
val_df['shift'] = 'val'
val_df.to_csv(f"{model_path}/{model_name}_val_layer_{layer}_{activation}_logsums.csv", index_label='id')

split = 'roll'
split_activations = pickle.load(open(f"{model_path}/{model_name}_{split}_layer_{layer}_{activation}_activation_ouputs.pkl", "rb"))

def activation_to_dataframe(activations, shift):
	mean_log_activations = mean_log_activation(activations)
	sum_mean_log_activations = np.sum(mean_log_activations, axis=1)
	df = pd.DataFrame(data=sum_mean_log_activations, columns=["log_sum"])
	df['shift'] = shift
	return df

def full_activation_to_dataframe(activations, shift):
	log_activations = np.log(activations).reshape(activations.shape[0], -1)
	df = pd.DataFrame(data=log_activations)
	df['shift'] = shift
	return df

split_dfs = [activation_to_dataframe(split_activations[i]['output'].numpy(), split_activations[i]['roll_pix']) for i in range(len(split_activations))]
full_split_dfs = [activation_to_dataframe(split_activations[i]['output'].numpy(), split_activations[i]['roll_pix']) for i in range(len(split_activations))]

pd.concat(split_dfs).to_csv(f"{model_path}/{model_name}_{split}_layer_{layer}_{activation}_logsums.csv", index_label='id')
pd.concat(split_dfs).to_csv(f"{model_path}/{model_name}_{split}_layer_{layer}_{activation}_logsums.csv", index_label='id')