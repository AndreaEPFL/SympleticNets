import torch
import torch.nn.functional as F
from utils import *

net4 = torch.nn.Sequential(
    torch.nn.Linear(4, 20),
    torch.nn.Tanh(),
    torch.nn.Linear(20, 40),
    torch.nn.Tanh(),
    torch.nn.Linear(40, 20),
    torch.nn.Tanh(),
    torch.nn.Linear(20, 4),
)


class ResNetLin(torch.nn.Module):
    # Residual Network for 2D data (Input, Output)

    def __init__(self, output_size, input_filters=32):
        super().__init__()
        self.inp_fil = input_filters
        self.fc1_out = 32

        self.lin1 = torch.nn.Linear(output_size, self.inp_fil)
        self.lin2 = torch.nn.Linear(self.inp_fil, self.inp_fil * 2)
        self.lin3 = torch.nn.Linear(self.inp_fil * 2, self.inp_fil * 4)

        self.fc1 = torch.nn.Linear(self.inp_fil * 4, self.fc1_out)
        self.fc2 = torch.nn.Linear(self.fc1_out, output_size)

    def forward(self, x):
        x1 = x.clone()

        x2 = torch.selu(self.lin1(x))
        x2 = torch.selu(self.lin2(x2))
        x2 = torch.selu(self.lin3(x2))
        x2 = torch.selu(self.fc1(x2))
        x2 = torch.selu(self.fc2(x2) + x1)

        return x2


class sympLinearA(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        mat2 = self.symmetric_matrix()
        mat3 = self.symmetric_matrix()

        self.weights2 = torch.nn.Parameter(mat2)  # Partial matrix this part only need to be updated
        self.weights3 = torch.nn.Parameter(mat3)
        self.bias = torch.nn.Parameter(torch.zeros(out_size))

    def forward(self, x):
        vect = self.weight_mul(x, self.weights2, True)
        vect = self.weight_mul(vect, self.weights3, False)

        return vect + self.bias

    def weight_mul(self, x, w, up=True):
        n = int(self.in_size / 2)
        p, q = torch.split(x, n)
        if up:
            p = p + torch.matmul(w, q)
        else:
            q = q + torch.matmul(w, p)

        return torch.cat((p, q))

    def symmetric_matrix(self):
        A = torch.Tensor(int(self.in_size / 2), int(self.out_size / 2)).type(torch.float32)
        A = (A + torch.transpose(A, 0, 1)) / 2

        return A


class activation_moduleA(torch.nn.Module):
    def __init__(self, N, act_function=F.selu):
        super().__init__()

        self.size = int(N / 2)  # N should be the number of variables
        self.weight = torch.nn.Parameter(torch.Tensor(self.size))
        self.function = act_function  # Activation function chose. Default is SELU

    def forward(self, x):
        p, q = torch.split(x, self.size)  # Divide input in (P, Q)
        q = torch.reshape(torch.diag(self.weight).mm(torch.reshape(self.function(p), (self.size, 1))) + \
                          torch.reshape(q, (self.size, 1)), (-1,))
        return torch.cat((p, q))


class SympNetA(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define all Layers Here
        self.lin1 = sympLinearA(4, 4)
        self.act = activation_module(4)

        # Activation function

    def forward(self, x):
        # Connect the layer Outputs here to define the forward pass
        x = self.act(self.lin1(x))

        return x


class sympLinear(torch.nn.Module):
    # Linear module
    def __init__(self, in_size, n_linear_layers):
        super().__init__()

        self.n = int(in_size / 2)
        self.out_size = in_size
        self.n_linear_layers = n_linear_layers  # Number of 'combined' linear modules in the same layer
        self.weights = self.init_weights()

    def forward(self, x):
        p, q = torch.split(x, self.n)
        for i in range(self.n_linear_layers):
            A = self.weights[str(i)]
            S = (A + A.t()) / 2
            if i & 1:
                # p = torch.transpose(torch.transpose(p, 0, 1) + S.mm(torch.transpose(q, 0, 1)), 0, 1)
                p = p + torch.matmul(S, q)
            else:
                # q = torch.transpose(torch.transpose(q, 0, 1) + S.mm(torch.transpose(p, 0, 1)), 0, 1)
                q = q + torch.matmul(S, p)
        return torch.cat((p, q)) + self.weights['bias']

    def init_weights(self):
        weights = torch.nn.ParameterDict()
        for i in range(self.n_linear_layers):
            weights[str(i)] = torch.nn.Parameter(torch.Tensor(self.n, self.n), requires_grad=True)
            torch.nn.init.xavier_uniform_(weights[str(i)])
        weights['bias'] = torch.nn.Parameter(xavier_initialization(torch.Tensor(self.out_size)), requires_grad=True)
        # weights['bias'] = torch.nn.Parameter(torch.Tensor(self.out_size), requires_grad=True)
        # torch.nn.init.xavier_uniform_(weights['bias'])
        # weights['bias'] = xavier_initialization(torch.nn.Parameter(torch.Tensor(self.out_size), requires_grad=True))
        return weights


class activation_module(torch.nn.Module):
    # Activation module
    def __init__(self, in_size, version, act_function=F.selu):
        super().__init__()

        self.size = int(in_size / 2)
        self.function = act_function
        self.weights_activation = self.init_weights()  # Activation function choice. Default is SELU
        self.version = version  # Select type of 'symplectic' activation (low or up)

    def forward(self, x):
        p, q = torch.split(x, self.size)  # Divide input in (P, Q)
        if self.version == 'low':
            q = torch.diag(self.weights_activation['a_scalar']) @ self.function(p) + q
        else:
            p = torch.diag(self.weights_activation['a_scalar']) @ self.function(q) + p
        return torch.cat((p, q))

    def init_weights(self):
        weights_activation = torch.nn.ParameterDict()
        weights_activation['a_scalar'] = torch.nn.Parameter(xavier_initialization(torch.Tensor(self.size)),
                                                            requires_grad=True)
        # torch.nn.init.xavier_uniform_(weights_activation['a_scalar'])
        # weights_activation['a_scalar'] = xavier_initialization(torch.nn.Parameter(torch.Tensor(self.size), requires_grad=True))
        return weights_activation


class SympNet(torch.nn.Module):
    # Network class
    def __init__(self, in_size, n_layers, n_linear_layers):
        super().__init__()

        self.in_size = in_size
        self.n = int(in_size / 2)
        self.n_layers = n_layers
        self.n_linear_layers = n_linear_layers
        self.weights = self.init_weights()

    def forward(self, x):
        for i in range(self.n_layers - 1):
            linear_part = self.weights['lin{}'.format(i)]
            nonlinear_part_low = self.weights['actlow{}'.format(i)]
            nonlinear_part_up = self.weights['actup{}'.format(i)]
            x = nonlinear_part_up(nonlinear_part_low(linear_part(x)))
        output_linear_layer = self.weights['linend']
        return output_linear_layer(x)

    def init_weights(self):
        layer_weights = torch.nn.ModuleDict()
        for i in range(self.n_layers - 1):
            layer_weights['lin{}'.format(i)] = sympLinear(self.in_size, self.n_linear_layers)
            layer_weights['actlow{}'.format(i)] = activation_module(self.in_size, 'low')
            layer_weights['actup{}'.format(i)] = activation_module(self.in_size, 'up')
        layer_weights['linend'] = sympLinear(self.in_size, self.n_linear_layers)
        return layer_weights
