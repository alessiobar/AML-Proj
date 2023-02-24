# Pytorch CIFAR-10 training configuration file
import math

start_epoch = 1
start_epoch_darbn = 161
num_epochs = 160
num_epochs_darbn = 40
batch_size = 128
optim_type = 'SGD'

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):

    optim_factor = 0
    if(epoch > 180):
        optim_factor = 2
    elif(epoch > 160):
        optim_factor = 1

    return init*math.pow(0.01, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
