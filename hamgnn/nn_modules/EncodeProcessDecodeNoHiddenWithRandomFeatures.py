import torch
import torch_geometric as torch_g
from hamgnn.nn_modules.EncodeProcessDecodeRandFeatures import EncodeProcessDecodeRandFeatures


class EncodeProcessDecodeNoHiddenRandomFeatures(EncodeProcessDecodeRandFeatures):
    def _construct_encoder_and_decoder(self):
        encoder_out_dim = self.hidden_dim - self.rand_features_dim
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(3, encoder_out_dim))
        decoder_nn = torch.nn.Sequential(torch.nn.Linear(encoder_out_dim + self.hidden_dim, self.out_dim))
        _mock_initial_h = torch.zeros_like(self.initial_h)
        del self.initial_h
        self.initial_h = _mock_initial_h
        return encoder_nn, decoder_nn

    def raw_next_step_logits(self, d: torch_g.data.Data):
        d.z = self.encoder_nn(d.x.clone())
        d.h = self.processor_nn(d.z, d.edge_index, d.edge_attr)
        return torch.squeeze(self.decoder_nn(torch.cat([d.z, d.h], dim=-1)), dim=-1)
