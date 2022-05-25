# write engine class containing train evaluate and inference methods here
import torch
import torch.nn as nn

from tqdm import tqdm


def train(dataset, data_loader, model, criterion, optimizer, device):
    """
    This function does training for one epoch
    :param data_loader: This is the pytorch dataloader
    :param model: pytorch model
    :param optimizer: optimizer, for e.g. adam, sgd, etc
    :param device: cuda/cpu
    """

    # put the model in train mode
    model.train()

    # calculate number of batches
    num_batches = int(len(dataset) / data_loader.batch_size)    
    
    count=0
    running_loss = 0.0
    # go over every batch of data in the data loader
    # for data in data_loader:
    # for data in tqdm(data_loader, total=num_batches):
    for data in tqdm(data_loader, total=len(data_loader)):
        # remember we have image and target in our dataset class
        inputs = data["x"]
        targets = data["y"]

        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        # zero grad the optimizer
        optimizer.zero_grad()
        # do the forward step of model
        output = model(inputs)
        # calculate loss
        loss = criterion(output, targets.view(-1,1))
        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        # if you have scheduler, you either need to step it here
        # or you have to step it after the epoch. here, we are not 
        # using any learning rate scheduler

        # print statistics
        running_loss += loss.item()

        if count==100:
            # break
            pass
        else:
            count+=1

    return running_loss



def evaluate(dataset, data_loader, model, criterion, device):
    """
    This function does evaluation for one epoch
    :param data_loader: this is the pytorch dataloader
    :param model: pytorch model
    :param device: cuda/cpu
    """

    # put model in evaluation mode
    model.eval()

    # calculate number of batches
    num_batches = int(len(dataset) / data_loader.batch_size)   

    # init lists to store targets and outputs
    final_targets = []
    final_outputs = []
    # init final_loss to 0
    final_loss = 0.0

    # we use no_grad context
    with torch.no_grad():
        count = 0
        for data in data_loader:
            inputs = data["x"]
            targets = data["y"]

            inputs = inputs.to(device,dtype=torch.float)
            targets = targets.to(device,dtype=torch.float)

            # do the forward pass
            output = model(inputs)

            # loss
            loss = criterion(output, targets.view(-1,1))

            # convert targets and outputs to lists
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            # extend the original list
            final_targets.extend(targets)
            final_outputs.extend(output)
            
            # add loss to final loss
            final_loss += loss

            if count==100:
                # break
                pass
            else:
                count+=1
        # avg loss
        avg_loss = final_loss / num_batches
    
    # return final output and final targets
    return final_outputs, final_targets, avg_loss
