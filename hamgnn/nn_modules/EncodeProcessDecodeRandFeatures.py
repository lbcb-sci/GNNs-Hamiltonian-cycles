import torch
from hamgnn.nn_modules.EncodeProcessDecodeNN import EncodeProcessDecodeAlgorithm
from hamgnn.nn_modules.nn_modules import RandomFeaturesWrapper, ResidualMultilayerMPNN


class EncodeProcessDecodeRandFeatures(EncodeProcessDecodeAlgorithm):
    def __init__(self, rand_features_dim=4, **kwargs):
        self.rand_features_dim = rand_features_dim
        super().__init__(**kwargs)

    def _construct_processor(self):
        return RandomFeaturesWrapper(
            ResidualMultilayerMPNN(dim=self.hidden_dim, message_dim=self.hidden_dim, edges_dim=1, nr_layers=self.processor_depth),
            self.rand_features_dim)

    def _construct_encoder_and_decoder(self):
        encoder_out_dim = self.hidden_dim - self.rand_features_dim
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(3 + self.hidden_dim, encoder_out_dim))
        decoder_nn = torch.nn.Sequential(torch.nn.Linear(encoder_out_dim + self.hidden_dim, self.out_dim))
        return encoder_nn, decoder_nn
