import torch
from torchvision import transforms as T
import tensorflow_datasets as tfds
import numpy as np 


corruptions = ['brightness', 'contrast', 'defocus_blur', 'elastic', 
               'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
              'frosted_glass_blur', 'impulse_noise', 'pixelate', 'saturate', 'shot_noise', 
              'spatter', 'speckle_noise', 'zoom_blur', ]

data_dir = "data/cifar10_c"

mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

for corruption in corruptions:
	for idx in range(1,6):
		ds, _ = tfds.load(f"cifar10_corrupted/{corruption}_{idx}", with_info=True, shuffle_files=False, data_dir=data_dir)
		np_imgs = np.stack([data['image']/255. for data in iter(tfds.as_numpy(ds)['test'])], axis=0)
		np_labels = np.stack([data['label'] for data in iter(tfds.as_numpy(ds)['test'])])
		transformed_imgs = torch.stack([transform(img) for img in np_imgs])
		np.save(f"{data_dir}/{corruption}_{idx}.npy", transformed_imgs.numpy())
		np.save(f"{data_dir}/{corruption}_{idx}_labels.npy", np_labels)