import torch
from ModelAE import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 200

# from os import listdir
# from os.path import join
# full_dataset=[]
# for f in listdir('../working_data')[:2]:
#     full_dataset+=list(np.load(join('../working_data',f))['arr_0'])
# random.shuffle(full_dataset)
# full_dataset=np.array(full_dataset)

full_dataset = (np.load('../generated_dataset.npz')['arr_0']+80)/80
print(full_dataset.shape)
train_size = int(0.85 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_dataset, test_dataset = torch.Tensor(train_dataset),torch.Tensor(test_dataset)

train_loader = DataLoader(TensorDataset(train_dataset, train_dataset), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_dataset, test_dataset), batch_size=batch_size, shuffle=True)

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.00135

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 32

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


num_epochs = 75
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
   train_loss =train_epoch(encoder,decoder,device,
   train_loader,loss_fn,optim)
   val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
   print('EPOCH {}/{} \t train loss {:.5f} \t val loss {:.5f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
   diz_loss['train_loss'].append(train_loss)
   diz_loss['val_loss'].append(val_loss)

torch.save(encoder.state_dict(), 'model_encoder.tor')
torch.save(decoder.state_dict(), 'model_decoder.tor')





# model = ConvAutoencoder().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
# criterion = nn.BCELoss()

# # Epochs
# n_epochs = 100
#
# for epoch in range(1, n_epochs + 1):
#     # monitor training loss
#     train_loss = 0.0
#
#     # Training
#     for data in train_loader:
#         images, _ = data
#         images = images.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, images)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * images.size(0)
#
#     train_loss = train_loss / len(train_loader)
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
