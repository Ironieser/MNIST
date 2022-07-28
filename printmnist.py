import torch
import PIL
train_data = r'./dataset_method_1/MNIST/processed/training.pt'
test_data = r'./dataset_method_1/MNIST/processed/test.pt'
train = torch.load(train_data)
imgs,gts = train
img1 = imgs[0]
gt = gts[0]