import torch
import torch_geometric as torch_g
import torch.nn as nn
from hamgnn.nn_modules.EncodeProcessDecodeNN import EncodeProcessDecodeAlgorithm


class EncodeProcessDecodeAdvancedPositionalEmbedding(EncodeProcessDecodeAlgorithm):
    def __init__(self, rand_features_dim=4, **kwargs):
        super().__init__(**kwargs)
        self.preencoder_net = nn.Sequential(nn.Linear(3, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.LayerNorm(self.hidden_dim))

    def _construct_encoder_and_decoder(self):
        _, decoder_nn = super()._construct_encoder_and_decoder()
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
        return encoder_nn, decoder_nn

    def raw_next_step_logits(self, d: torch_g.data.Data):
        d.z = self.encoder_nn(self.preencoder_net(d.x.clone()) + d.h)
        d.h = self.processor_nn(d.z, d.edge_index, d.edge_attr)
        return torch.squeeze(self.decoder_nn(torch.cat([d.z, d.h], dim=-1)), dim=-1)
