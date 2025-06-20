import torch 

def train_network(network, dataloader, device, lr = 0.001, epochs=100):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (input_batch, output_batch) in enumerate(dataloader):
            
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            optimizer.zero_grad()
            output = network(input_batch)
            loss = criterion(output, output_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Basic Data Loss: {loss.item()}')
    return loss.item()  # Return the loss for the last epoch