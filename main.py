from utils import *
from networks import *
import torch.utils.data as Data
import time
import wandb


def main():
    # wandb.ai configuration
    wandb.init(project="sympletic_project_report")

    # Parameters

    config = wandb.config
    config.lr = 0.001
    config.architecture = "SympNet"
    config.batch_size = 200
    config.epochs = 40
    config.dataset = "Lotka_Volterra_(0.4,-0.7)"
    config.layer_number = 2
    config.layer_length = 2

    # Import data generated on Matlab
    pathLV = r"semester_project_material\Lotka_Volterra_problem\data\LV_a_1.2_b_1_c_1_d_2_init_pred_0.4_init_prey_-0.7_Integrator_Symplectic_Euler.txt"
    pathKepler = r'semester_project_material\Kepler_problem\2D\data\KP1D_M_1_init_dis_1.0429_Integrator_Stormer_Verlet.txt'
    data = load_data(pathLV)
    data = data[0:12000]

    # Validation set from 30 to 40 sec.
    data_val = load_data(pathLV)
    data_val = data_val[12000:16000]

    # Create the time vector
    time_vector = create_time_vector(30, 12000)

    """"# Data visualisation
    plot_data(data[:, 0:2], time_vector, 15, 6, 'Time (s)', 'Velocity', 'Velocity of the elements')
    plot_data(data[:, 2:4], time_vector, 15, 6, 'Time (s)', 'Position', 'Position of the elements on the X-axis')"""

    # Data visualisation
    plot_data(data, time_vector, 15, 6, 'Time (s)', 'Value', 'Value of the elements - LV')

    # Create variables
    net_input, net_output = net_variables(data)

    # Network used
    smpNet = SympNet(2, 2, 2)  # nb variables, nb layers, length of layer
    # resnet = ResNetLin(4)
    # net = net4

    net = smpNet
    # Initialize data
    net_input2 = data_process(net, net_input)

    # Define optimizer and loss function for the network training
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # Mean squared loss

    BATCH_SIZE = 200
    EPOCH = 40

    torch_dataset = Data.TensorDataset(net_input2, net_output)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE)

    start_time = time.time()  # Chronometer time

    loss_store = []  # Store the losses to plot them afterwards

    # Start training
    for epoch in range(EPOCH):

        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # with torch.autograd.set_detect_anomaly(True):
            prediction = data_process(net, batch_x)  # input x and prediction based on x
            net.zero_grad()  # Clear gradients for the next train #optimizer.zero_grad
            loss = loss_func(prediction, batch_y)  # compute the loss
            loss.backward(retain_graph=True)  # backpropagation, compute gradients
            optimizer.step()

        loss_store.append(loss.item())
        wandb.log({"loss": loss.item()})
        print(f"Epoch: {epoch}, Training loss: {loss.item()}")

    # Plot the losses
    fig = plt.figure(figsize=(15, 9), dpi=100)

    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))

    # Plot de results for LV
    prediction = data_process(net, net_input)
    new_time_vector = time_vector[1:len(time_vector)]
    fig = plt.figure(figsize=(30, 20), dpi=100)

    plt.subplot(2, 1, 1)
    plt.plot(new_time_vector, data[1:len(time_vector), 0], label='Solutions')
    plt.plot(new_time_vector, prediction.data.numpy()[:, 0], '--', label='Net Prediction')
    plt.title("Prediction vs Data Plot")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(new_time_vector, data[1:len(time_vector), 1], label='Solutions')
    plt.plot(new_time_vector, prediction.data.numpy()[:, 1], '--', label='Net Prediction')
    plt.title("Prediction vs Data Plot")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")

    fig.savefig('Prediction vs Data - LV - SympNet- (2, 2)', bbox_inches='tight', dpi=100)

    plt.show()

    """# Plot the results
    prediction = data_process(net, net_input)
    new_time_vector = time_vector[1:len(time_vector)]
    fig = plt.figure(figsize=(30, 20), dpi=100)

    plt.subplot(2, 2, 1)

    plt.plot(new_time_vector, data[1:len(time_vector), 0], label='Solutions')
    plt.plot(new_time_vector, prediction.data.numpy()[:, 0], '--', label='Net Prediction')
    plt.title("Prediction vs Data Plot")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity - Left ball")
    plt.legend()

    plt.subplot(2, 2, 2)

    plt.plot(new_time_vector, data[1:len(time_vector), 1], label='Solutions')
    plt.plot(new_time_vector, prediction.data.numpy()[:, 1], '--', label='Net Prediction')
    plt.title("Prediction vs Data Plot")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity - Right ball")
    plt.legend()

    plt.subplot(2, 2, 3)

    plt.plot(new_time_vector, data[1:len(time_vector), 2], label='Solutions')
    plt.plot(new_time_vector, prediction.data.numpy()[:, 2], '--', label='Net Prediction')
    plt.title("Prediction vs Data Plot")
    plt.xlabel("Time (s)")
    plt.ylabel("Position - Left ball")
    plt.legend()

    plt.subplot(2, 2, 4)

    plt.plot(new_time_vector, data[1:len(time_vector), 3], label='Solutions')
    plt.plot(new_time_vector, prediction.data.numpy()[:, 3], '--', label='Net Prediction')
    plt.title("Prediction vs Data Plot")
    plt.xlabel("Time (s)")
    plt.ylabel("Position - Right ball")
    plt.legend()

    fig.savefig('Prediction vs Data - Kepler - SympNet- (2, 10)', bbox_inches='tight', dpi=100)

    plt.show()"""

    validation_qty = validation_model(data_val, net, loss_func)
    print("The MSE for the validation set is equal to", validation_qty)
    wandb.log({"Validation loss": validation_qty})
    wandb.finish()

    print(type(loss), type(prediction))
    return None


if __name__ == '__main__':
    main()
