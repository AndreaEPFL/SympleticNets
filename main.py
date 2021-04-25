from utils import *
from networks import *
import torch.utils.data as Data
import time
import wandb


def main():
    # wandb.ai configuration

    wandb.init(project="sympletic_project")

    # Parameters

    config = wandb.config
    config.lr = 0.001
    config.architecture = "SympNet"
    config.batch_size = 400
    config.epochs = 300
    config.dataset = "Kepler_init_dis_1.0429"

    # Import data generated on Matlab
    data = load_data(
        r'semester_project_material\Kepler_problem\2D\data\KP1D_M_1_init_dis_1.0429_Integrator_Stormer_Verlet.txt')
    data = data[0:12000]

    # Validation set from 30 to 40 sec.
    data_val = load_data(
        r'semester_project_material\Kepler_problem\2D\data\KP1D_M_1_init_dis_1.0429_Integrator_Stormer_Verlet.txt')
    data_val = data_val[12000:16000]

    # Create the time vector
    time_vector = create_time_vector(30, 12000)

    # Data visualisation
    #plot_data(data[:, 0:2], time_vector, 15, 6, 'Time (s)', 'Velocity', 'Velocity of the elements')
    #plot_data(data[:, 2:4], time_vector, 15, 6, 'Time (s)', 'Position', 'Position of the elements on the X-axis')

    # Create variables
    net_input, net_output = net_variables(data)

    # Network used
    smpNet = SympNet()
    resnet = ResNetLin(4)
    net = net4

    # Initialize data
    net_input = data_process(smpNet, net_input)

    # Define optimizer and loss function for the network training
    optimizer = torch.optim.Adam(smpNet.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()  # Regression mean squared loss

    BATCH_SIZE = 400
    EPOCH = 50

    torch_dataset = Data.TensorDataset(net_input, net_output)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)

    start_time = time.time()  # Chronometer time

    loss_store = []  # Store the losses to plot them afterwards

    # Start training
    for epoch in range(EPOCH):

        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            with torch.autograd.set_detect_anomaly(True):
                prediction = data_process(smpNet, batch_x)  # input x and prediction based on x

            # for name, param in resNet.named_parameters():
            #    if param.requires_grad:
            #        print(name, param.data)

                loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)
                optimizer.zero_grad()  # clear gradients for next train
                loss.backward(retain_graph=True)  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            # Test sympleticity
            #print("Sympletic test :", sympletic_test(4, b_x, prediction))

        loss_store.append(loss.item())
        wandb.log({"loss": loss.item()})
        print(f"Epoch: {epoch}, Training loss: {loss.item()}")


    # Plot the losses
    fig = plt.figure(figsize=(15, 9), dpi=100)

    plt.grid()
    plt.plot(np.linspace(0, EPOCH, EPOCH), loss_store)
    plt.title("Loss - sNet")

    fig.savefig('Loss sNet - Kepler', bbox_inches='tight', dpi=100)

    plt.show()

    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))

    # Finish wandb
    wandb.finish()

    # Plot the results
    prediction = data_process(smpNet, net_input)
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

    fig.savefig('Prediction vs Data - Kepler - SympNet', bbox_inches='tight', dpi=100)

    plt.show()

    #validation_qty = validation_model(data_val, resnet, loss_func)
    #print("The MSE for the validation set is equal to", validation_qty)

    print(type(loss), type(prediction))
    return None


if __name__ == '__main__':
    main()
