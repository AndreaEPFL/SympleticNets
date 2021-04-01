import torch
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import time


def load_data(path):
    data = np.loadtxt(path, delimiter=",")  # Use r ?
    return data


def create_time_vector(stop, steps):
    t = np.linspace(0, stop, steps)
    return t


def plot_data(data, time_vector, size_x, size_y, x_label, y_label, title):
    fig = plt.figure(figsize=(size_x, size_y), dpi=100)

    plt.scatter(time_vector, data[:, 0], c='b', marker='.', label='Left el.', linewidths=0.7)
    plt.scatter(time_vector, data[:, 1], c='r', marker='.', label='Right el.', linewidths=0.7)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    fig.savefig(title, bbox_inches='tight', dpi=100)

    plt.show()
    return None


def net_variables(data):
    data_len = len(data)

    in_var = Variable(torch.from_numpy(data[0:(data_len - 1)]).type(torch.float32))
    out_var = Variable(torch.from_numpy(data[1:(data_len + 1)]).type(torch.float32))

    return in_var, out_var


def validation_model(data, network, loss):
    dat = Variable(torch.from_numpy(data).type(torch.float32))  # Take values from 10 to 15 sec. only
    prediction = network(dat)
    loss_quantity = loss(dat, prediction)

    return loss_quantity.item()


def sympletic_test(N, grad):
    n = int(N / 2)
    J_np = np.append(np.zeros((n, n)), np.identity(n), axis=1)
    J_np_bis = np.append((-1)*np.identity(n), np.zeros((n,n)), axis=1)

    J = torch.from_numpy(np.append(J_np, J_np_bis, axis=0)).type(torch.float32)

    test = torch.transpose(grad, 0, 1) @ J @ grad

    return torch.eq(J, test)