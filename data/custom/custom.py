import os
import argparse

ls = os.listdir('./images')

train = open('./train.txt', 'w')
valid = open('./valid.txt', 'w')

train_path = 'data/images/'
valid_path = 'data/images/'

for l in ls:
    item=train_path+l
    train.writelines(item+'\n')
    valid.writelines(item+'\n')
