from utils import *
from networks import *
import torch.utils.data as Data
import pandas as pd


def main():
    # Import data generated on Matlab
    data = load_data(r'2D/data/KP1D_M_1_init_dis_1.0429_Integrator_Stormer_Verlet.txt')
    data = data[0:12000]

    # Validation set from 30 to 40 sec.
    data_val = load_data(r'2D/data/KP1D_M_1_init_dis_1.0429_Integrator_Stormer_Verlet.txt')
    data_val = data_val[12000:16000]

    # Create the time vector
    time_vector = create_time_vector(30, 12000)

    # Create variables
    net_input, net_output = net_variables(data)

    # Use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Network used
    smpNet = SympNet(4, 10, 10)

    # Initialize data
    net_input = data_process(smpNet, net_input)

    smpNet.cuda(device=device)

    # Test if all parameters are on cuda device
    for name, param in smpNet.named_parameters():
        if param.device.type != 'cuda':
            print(name, param.device.type)

    # Define optimizer and loss function for the network training
    optimizer = torch.optim.Adam(smpNet.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # Regression mean squared loss

    BATCH_SIZE = 400
    EPOCH = 500

    torch_dataset = Data.TensorDataset(net_input, net_output)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE)

    loss_store = []  # Store the losses

    # Start training
    for epoch in range(EPOCH):

        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step

            b_x = batch_x.to(device)
            b_y = batch_y.to(device)

            prediction = data_process(smpNet, b_x)  # input x and predict based on x
            print("Prediction on cuda",  prediction.is_cuda)

            loss = loss_func(prediction, b_y)  # must be (1. nn output, 2. target)
            print("Loss on cuda", loss.is_cuda)

            optimizer.zero_grad()  # clear gradients for next train
            loss.register_hook(lambda grad: print(grad))
            print(torch.cuda.current_device())
            print(torch.cuda.device(0))
            print(torch.cuda.get_device_name(0))

            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        loss_store.append(loss.item())
        print(f"Epoch: {epoch}, Training loss: {loss.item()}")

    prediction = data_process(smpNet, net_input)

    dat_val = data_val.to(device)
    validation_qty = validation_model(dat_val, smpNet, loss_func)
    print("The MSE for the validation set is equal to", validation_qty)

    # Extract losses
    df_loss = pd.DataFrame(loss.numpy())
    df_loss.to_csv('Loss_SympNet.csv', index=False)

    # Extract Prediction
    df_pred = pd.DataFrame(prediction.numpy())
    df_pred.to_csv('Pred_SympNet.csv', index=False)

    return None


if __name__ == '__main__':
    main()
