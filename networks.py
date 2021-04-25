import torch
import torch.nn.functional as F

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
        x1 = x

        x = torch.selu(self.lin1(x))
        x = torch.selu(self.lin2(x))
        x = torch.selu(self.lin3(x))
        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x) + x1)

        return x


class sympLinear(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        mat2 = self.symmetric_matrix()
        mat3 = self.get_sympletic_app()

        self.weights2 = torch.nn.Parameter(mat2)  # Partial matrix this part only need to be updated
        self.weights3 = torch.nn.Parameter(mat3)  # Not used for the moment
        self.bias = torch.nn.Parameter(torch.zeros(out_size))

    def forward(self, x):
        n = int(self.in_size / 2)
        p, q = torch.split(x, n)
        p = torch.reshape(torch.reshape(p, (n, 1)) +
                            (self.weights2 + torch.transpose(self.weights2, 0, 1)).mm(torch.reshape(q, (n, 1))), (-1,))

        return torch.cat((p, q)) + self.bias

    def get_sympletic_app(self):
        in_shape = int(self.in_size / 2)
        out_shape = int(self.out_size / 2)

        A = torch.Tensor(in_shape, out_shape).type(torch.float32)
        A = (A + torch.transpose(A, 0, 1)) / 2

        S = torch.cat((torch.diag(torch.ones(out_shape)), A), 1)
        S_ = torch.cat((torch.zeros(in_shape, out_shape), torch.diag(torch.ones(out_shape))), 1)

        return torch.cat((S, S_), 0)

    def symmetric_matrix(self):
        A = torch.Tensor(int(self.in_size / 2), int(self.out_size / 2)).type(torch.float32)
        A = (A + torch.transpose(A, 0, 1)) / 2

        return A


class activation_module(torch.nn.Module):
    def __init__(self, N, act_function=F.selu):
        super().__init__()

        self.size = int(N / 2)  # N should be the number of variables
        self.weight = torch.nn.Parameter(torch.Tensor(self.size))
        self.function = act_function  # Activation function chose. Default is SELU

    def forward(self, x):
        p, q = torch.split(x, self.size, dim=1)  # Divide input in (P, Q)
        q = torch.diag(self.weight) @ self.function(p) + q
        return torch.cat((p, q), 1)


class SympNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define all Layers Here
        self.lin1 = sympLinear(4, 4)

        # Activation function

    def forward(self, x):
        # Connect the layer Outputs here to define the forward pass
        x = self.lin1(x)

        return x
