import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp

from deepface import DeepFace
import pandas as pd
import os

def run():
	test_opts = TestOptions().parse()

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	if 'learn_in_w' not in opts:
		opts['learn_in_w'] = False
	if 'output_size' not in opts:
		opts['output_size'] = 1024

	opts = Namespace(**opts)

	net = pSp(opts)
	net.eval()
	net.cuda()

	print('Loading dataset for {}'.format(opts.dataset_type))

	dataset_args = data_configs.DATASETS[opts.dataset_type]
	transforms_dict = dataset_args['transforms'](opts).get_transforms()
	dataset = InferenceDataset(root=opts.data_path,
	                           transform=transforms_dict['transform_inference'],
	                           opts=opts)

	if opts.n_images is None:
		opts.n_images = len(dataset)

	if opts.MODE == 'random':
		result_csv=pd.DataFrame()
		mask = list(pd.read_csv('True_data.csv')['input'])
		count=1

		for j in range(100):
			if int(dataset.paths[j].split('/')[1].split('.')[0]) in mask:
				index = int(dataset.paths[j].split('/')[1].split('.')[0])
			else:
				continue

			x_input = dataset[j].cuda()
			x_input.unsqueeze_(0).to("cuda").float()

			y_input = torch.from_numpy(np.random.randn(1,512).astype('float32')).to("cuda")
			y_input = net(y_input, input_code=True)

			latent_mask = [x for x in range(18)]

			if not os.path.exists('/home/ljj/pixel2style2pixel/result/'+str(index)):
				os.makedirs('/home/ljj/pixel2style2pixel/result/'+str(index))

			for i in range(18):
				del latent_mask[0]
				res = net(x=x_input, y=y_input, latent_mask=latent_mask, resize=opts.resize_outputs, MODE=opts.MODE)
				res = tensor2im(res.squeeze(0))
				res.save('result/' + str(index) + '/' + str(i) + '.jpg')

			result = []

			for i in range(18):
				result.append(DeepFace.verify(img1_path='/home/ljj/pixel2style2pixel/datas/'+str(index)+'.jpg', img2_path='/home/ljj/pixel2style2pixel/result/'+str(index)+'/'+str(i)+'.jpg',enforce_detection=False))

			result_verified = []
			result_distance = []

			for i in range(18):
				result_verified.append(str(result[i]['verified']))
				result_distance.append(result[i]['distance'])

			result = pd.DataFrame(data={'verified':result_verified,'distance':result_distance},index=[x for x in range(18)])
			result.columns =[[str(index)+'image',str(index)+'image'],['verified','distance']]

			if count==1:
				result_csv = result
				count = count + 1
			else:
				result_csv = pd.concat([result_csv, result],axis=1)

		result_csv.to_csv('/home/ljj/pixel2style2pixel/result/verify.csv')


	elif opts.MODE == 'cross':
		x_input = dataset[0].cuda()
		y_input = dataset[1].cuda()

		x_input.unsqueeze_(0).to("cuda").float()
		y_input.unsqueeze_(0).to("cuda").float()

		latent_mask = []
		for i in range(18):
			latent_mask.append(i)
			res = net(x=x_input, y=y_input, latent_mask=latent_mask, resize=opts.resize_outputs, MODE=opts.MODE)
			res = tensor2im(res.squeeze(0))
			res.save('result/result'+str(i)+'.jpg')

		latent_mask = []
		for i in range(18):
			latent_mask.append(i)
			res = net(x=y_input, y=x_input, latent_mask=latent_mask, resize=opts.resize_outputs, MODE=opts.MODE)
			res = tensor2im(res.squeeze(0))
			res.save('result/result_reverse' + str(i) + '.jpg')

	elif opts.MODE == 'encoding':
		for i in range(len(dataset)):
			x_input = dataset[i].cuda()
			x_input.unsqueeze_(0).to("cuda").float()
			res = net(x=x_input, resize=opts.resize_outputs, MODE=opts.MODE)
			res = tensor2im(res.squeeze(0))
			res.save('encoding/'+dataset.paths[i].split('/')[1].split('.')[0]+'.jpg')

if __name__ == '__main__':
	run()
