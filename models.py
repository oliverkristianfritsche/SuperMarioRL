import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import random



class BaseNNModule(nn.Module):
    def __init__(self):
        super(BaseNNModule, self).__init__()

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def normalize_input(self, x):
        return x.float() / 255.0

class Autoencoder(BaseNNModule):
    def __init__(self, encoder_layers=None, decoder_layers=None):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_layers(encoder_layers or [
            {'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'in_channels': 32, 'out_channels': 64, 'kernel_size': 7, 'stride': 1, 'padding': 0}
        ], layer_type='encoder')

        self.decoder = self.build_layers(decoder_layers or [
            {'in_channels': 64, 'out_channels': 32, 'kernel_size': 7, 'stride': 1, 'padding': 0, 'output_padding': 0},
            {'in_channels': 32, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1},
            {'in_channels': 16, 'out_channels': 3, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        ], layer_type='decoder')

    def build_layers(self, layer_configs, layer_type='encoder'):
        layers = []
        for config in layer_configs:
            if layer_type == 'encoder':
                layers.append(nn.Conv2d(**config))
            else: 
                layers.append(nn.ConvTranspose2d(**config))
            layers.append(nn.ReLU())
        if layer_type == 'decoder':
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AutoEncoderResnetDQN(BaseNNModule):
    def __init__(self, input_shape, n_actions, feature_extractor_model=models.resnet18, pretrained=True):
        super(AutoEncoderResnetDQN, self).__init__()
        input_shape = (input_shape[2], input_shape[0], input_shape[1])  # Rearrange to (C, H, W)
        input_shape = (input_shape[0], input_shape[1] // 2, input_shape[2] // 2)
        self.encoder = Autoencoder()
        self.input_dim = np.prod(input_shape)
        self.feature_extractor = feature_extractor_model(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        self.max_pool = nn.MaxPool2d(2, 2)
        self.freeze_layers(self.feature_extractor)
        self.fc_layers = self.build_fc_layers(input_shape, n_actions)

        self.apply(self.init_weights)

    def freeze_layers(self, layers):
        for param in layers.parameters():
            param.requires_grad = False

    def build_fc_layers(self, input_shape, n_actions):
        with torch.no_grad():
            output_size = self._get_conv_output(input_shape)
            print(f'Output size: {output_size}')
        return nn.Sequential(
            nn.Linear(10752, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        # output = self.feature_extractor(dummy_input)
        output = self.max_pool(dummy_input)
        output = self.encoder(output)[0]  # Assuming you want to pass through the encoder first
        output_size = output.view(1, -1).size(1)
        return output_size


    def forward(self, x):
        x = self.normalize_input(x)
        x = self.max_pool(x)
        x = self.encoder(x)[0]
        # x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class EncoderResnetDQNWithValues(nn.Module):
    def __init__(self, input_shape, n_actions, feature_extractor_model=models.resnet18, pretrained=True):
        super(EncoderResnetDQNWithValues, self).__init__()
        # Rearrange input_shape to match PyTorch's (C, H, W)
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
        # Adjust input_shape if necessary, here assuming halved dimensions
        input_shape = (input_shape[0], input_shape[1] // 2, input_shape[2] // 2)
        
        self.feature_extractor = feature_extractor_model(pretrained=pretrained)
        # Modify to remove the last two layers
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        self.max_pool = nn.MaxPool2d(2, 2)
        self.freeze_layers(self.feature_extractor)
        
        # Output size calculation for fully connected layer
        with torch.no_grad():
            output_size = self._get_conv_output(input_shape)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),  # Single value output
        )

    def freeze_layers(self, layers):
        for param in layers.parameters():
            param.requires_grad = False

    def _get_conv_output(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.feature_extractor(dummy_input)
            return int(np.prod(output.size()))

    def forward(self, x):
        x = self.max_pool(x)
        x = self.feature_extractor(x)
        # Use .reshape() instead of .view() to avoid issues with non-contiguous tensors
        x = x.reshape(x.size(0), -1)
        action_scores = self.action_head(x)
        state_value = self.value_head(x).squeeze(-1)
        return action_scores, state_value