import torch
import torch.nn as nn
from base_module import BaseNNModule

class Autoencoder(BaseNNModule):
    def __init__(self, input_shape, encoder_layers=None, decoder_layers=None):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        
        # Debugging print statements
        print("Encoder layers input:", encoder_layers)
        print("Decoder layers input:", decoder_layers)

        self.encoder = self.build_layers(encoder_layers or self.default_encoder_layers(input_shape), layer_type='encoder')
        self.decoder = self.build_layers(decoder_layers or self.default_decoder_layers(input_shape), layer_type='decoder')

        self.apply(self.init_weights)

    def default_encoder_layers(self, input_shape):
        return [
            {'in_channels': input_shape[2], 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}  # Adjusted kernel_size and stride
        ]

    def default_decoder_layers(self, input_shape):
        return [
            {'in_channels': 64, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1},
            {'in_channels': 32, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1},
            {'in_channels': 16, 'out_channels': input_shape[2], 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}  # Adjusted to match input channels
        ]

    def build_layers(self, layer_configs, layer_type='encoder'):
        layers = []
        for config in layer_configs:
            if layer_type == 'encoder':
                layers.append(nn.Conv2d(**config))
            else:  # For 'decoder'
                layers.append(nn.ConvTranspose2d(**config))
            layers.append(nn.ReLU())
        if layer_type == 'decoder':
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

