import torch
train_data = r'./dataset_method_1/MNIST/processed/training.pt'
test_data = r'./dataset_method_1/MNIST/processed/test.pt'
rand_train = r'./dataset_method_1/MNIST/processed/rm_training.pt'
rand_test = r'./dataset_method_1/MNIST/processed/rm_test.pt'
train_data = torch.load(train_data)
test_data = torch.load(test_data)
import numpy as np
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed_all(0)

import random
y1 = torch.randint(0,10,(60000,))
y2 = torch.randint(0,10,(10000,))

y1 = torch.LongTensor(y1)
y2 = torch.LongTensor(y2)
# train_data[1] = y1
# train_data[1] = y1
# test_data[1] = y2
torch.save((train_data[0],y1),rand_train)
torch.save((test_data[0],y2),rand_test)
print(test_data)