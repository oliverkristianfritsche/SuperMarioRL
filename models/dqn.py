import torch
import torch.nn as nn
from autoencoder import Autoencoder
from base_module import BaseNNModule
import torch.nn.functional as F
class EncoderWithHead(Autoencoder):
    def __init__(self, input_shape,output_dims, encoder_layers=None, decoder_layers=None, head_layers_config=None):
        super(EncoderWithHead, self).__init__(input_shape, encoder_layers, decoder_layers)
        print("EncoderWithHead input shape:", input_shape, "output_dims:", output_dims)
        adjusted_input_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.flat_output_size = self.calculate_flat_output_size(adjusted_input_shape)
        
        self.head = None
        if head_layers_config:
            self.head = self.build_head(head_layers_config)
        else:
            self.head = default_head_layers_config = [
            {'out_features': 512},
            {'out_features': 256},  # Intermediate layer
            {'out_features': output_dims}
            ]

        self.output_dims = output_dims  
        self.head = self.build_head(default_head_layers_config, self.flat_output_size)

    def calculate_flat_output_size(self, input_shape):
        dummy_input = torch.rand(1, *input_shape)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        return int(torch.numel(dummy_output) / dummy_output.shape[0])
    
    def build_head(self, layer_configs, flat_output_size):
        layers = [nn.Flatten(start_dim=1)]  # Flatten all dimensions except batch
        in_features = flat_output_size
        for config in layer_configs[:-1]:
            out_features = config['out_features']
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features  # Update in_features for the next layer
        
        # Add the final output layer
        layers.append(nn.Linear(in_features, self.output_dims))  
        return nn.Sequential(*layers)

        
    def forward(self, x):
        encoded, decoded = super().forward(x)  # Use the Autoencoder's forward method
        if self.head:
            head_output = self.head(encoded)  # Process encoded features through the head
            # print("Head output:", head_output.shape)
            return head_output  # This should be of shape [batch_size, num_actions]
        else:
            raise NotImplementedError("Head not implemented in the network")


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        
        # Calculate the size of the output from the second convolutional layer
        # Assuming square input and no padding, directly use the new calculations:
        convw = convh = 9  
        
        linear_input_size = convw * convh * 32  # This should now be positive and correct
        
        self.fc1 = nn.Linear(linear_input_size, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        return self.out(x)
    
class DQN2(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[2], out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        convw = convh = 7  # Adjust this based on your specific input shape and conv layer parameters

        linear_input_size = convw * convh * 64  # This should now be positive and correct

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        return self.out(x)

