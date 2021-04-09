import torch
import numpy as np

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

        mat = self.get_sympletic_app()
        mat2 = self.symmetric_matrix()
        mat3 = self.get_sympletic_app()

        self.weights1 = torch.nn.Parameter(mat, requires_grad=True)  # 'requires_grad' to store gradient
        self.weights2 = torch.nn.Parameter(mat2, requires_grad=True)  # Partial matrix
        self.weights3 = torch.nn.Parameter(mat3)  # Not used for the moment
        self.bias = torch.nn.Parameter(torch.zeros(out_size))

    def forward(self, x):
        n = int(self.in_size/2)
        p, q = torch.split(x, n, dim=1)
        p = torch.transpose(torch.transpose(p, 0, 1) + self.weights2.mm(torch.transpose(q, 0, 1)), 0, 1)

        return torch.cat((p, q), 1) + self.bias

    def get_sympletic_app(self):
        rng = np.random.default_rng()
        sze = (int(self.in_size / 2), int(self.out_size / 2))
        A = rng.random(sze)
        A = (A + A.transpose()) / 2

        S = np.append(np.identity(int(self.out_size / 2)), A, axis=1)
        S_ = np.append(np.zeros(sze), np.identity(int(self.out_size / 2)), axis=1)

        return torch.from_numpy(np.append(S, S_, axis=0)).type(torch.float32)

    def symmetric_matrix(self):
        rng = np.random.default_rng()
        sze = (int(self.in_size / 2), int(self.out_size / 2))
        A = rng.random(sze)
        A = (A + A.transpose()) / 2

        return torch.from_numpy(A).type(torch.float32)


class SympNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define all Layers Here
        self.lin1 = sympLinear(4, 4)

    def forward(self, x):
        # Connect the layer Outputs here to define the forward pass

        x = self.lin1(x)

        return x