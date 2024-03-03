import torch
import torch.nn as nn
from autoencoder import Autoencoder
from base_module import BaseNNModule

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
