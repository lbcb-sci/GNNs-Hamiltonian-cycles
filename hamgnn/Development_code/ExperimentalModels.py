import torch
import torch_geometric as torch_g
import itertools
import os
import torchinfo

from hamgnn.NN_modules import MultilayerGatedGCN
from Experimental_NN_modules import ResidualMultilayerMPNN3DeepMessages, ResidualMultilayerAttentionMPNNWithEdgeFeatures
from hamgnn.Models import EncodeProcessDecodeAlgorithm, WalkUpdater, DEVICE, MODEL_WEIGHTS_FOLDER


class EncodeProcessDecodeWithDeepMessagesAlgorithm(EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return ResidualMultilayerMPNN3DeepMessages(dim=self.hidden_dim, message_dim=self.hidden_dim,
                                                   edges_dim=1, nr_layers=self.processor_depth)


class GatedGCNEncodeProcessDecodeAlgorithm(EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return MultilayerGatedGCN(dim=self.hidden_dim, nr_layers=self.processor_depth)

    def next_step_logits(self, d: torch_g.data.Data):
        d.z = self.encoder_nn(torch.cat([d.x, d.h], dim=-1))
        d.h, d.edge_attr = self.processor_nn(d.z, d.edge_index, d.edge_attr)
        return torch.squeeze(self.decoder_nn(torch.cat([d.z, d.h], dim=-1)), dim=-1)

    def init_graph(self, d):
        d.edge_attr = torch.ones([d.edge_index.shape[1]], device=self.device)
        d.h = torch.stack([self.initial_h for _ in range(d.num_nodes)], dim=-2)


class EncodeProcessDecodeWithEdgeFeatures(EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return ResidualMultilayerAttentionMPNNWithEdgeFeatures(
            node_dimension=self.hidden_dim, edge_dimension=self.edge_dim, nr_layers=self.processor_depth)

    def _construct_encoder_and_decoder(self):
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(3 + self.hidden_dim, self.hidden_dim))
        decoder_nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + self.hidden_dim, self.out_dim))
        return encoder_nn, decoder_nn

    def __init__(self, is_load_weights=True, processor_depth=3, in_dim=1, out_dim=1, hidden_dim=32, edge_dim=16,
                 device=DEVICE, graph_updater=WalkUpdater()):
        self.edge_dim = edge_dim
        super(EncodeProcessDecodeWithEdgeFeatures, self).__init__(False, processor_depth, in_dim, out_dim, hidden_dim, device, graph_updater)
        self.INITIAL_EDGE_TENSOR_NAME = f"{self.__class__.__name__}_EdgeAttributes.pt"
        self.initial_edge_attr = torch.rand(self.edge_dim, device=self.device)
        if is_load_weights:
            self.load_weights()

    def get_weights_paths(self, directory):
        return [os.path.join(directory, name) for name in
                [self.ENCODER_NAME, self.DECODER_NAME, self.PROCESSOR_NAME, self.INITIAL_HIDDEN_TENSOR_NAME,
                 self.INITIAL_EDGE_TENSOR_NAME]]

    def load_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        encoder_path, decoder_path, processor_path, initial_hidden_path, initial_edge_attr_path = self.get_weights_paths(
            directory)
        for module, path in zip(
                [self.encoder_nn, self.decoder_nn, self.processor_nn],
                [encoder_path, decoder_path, processor_path]):
            if os.path.isfile(path):
                module.load_state_dict(torch.load(path))
                print("Loaded {} from {}".format(module.__class__.__name__, path))
        if os.path.isfile(initial_hidden_path):
            self.initial_h = torch.load(initial_hidden_path)
            self.initial_h = self.initial_h.to(self.device)
        else:
            self.initial_h = torch.rand(self.hidden_dim, device=self.device)
        if os.path.isfile(initial_edge_attr_path):
            self.initial_edge_attr = torch.load(initial_edge_attr_path)
            self.initial_edge_attr = self.initial_h.to(self.device)

    def parameters(self):
        return itertools.chain(self.encoder_nn.parameters(), self.decoder_nn.parameters(),
                               self.processor_nn.parameters(), [self.initial_h, self.initial_edge_attr])

    def next_step_logits(self, d: torch_g.data.Data):
        d.z = self.encoder_nn(torch.cat([d.x, d.h], dim=-1))
        d.h, d.edge_attr = self.processor_nn(d.z, d.edge_attr, d.edge_index)
        return torch.squeeze(self.decoder_nn(torch.cat([d.z, d.h], dim=-1)), dim=-1)

    def init_graph(self, d):
        d.edge_attr = torch.stack([self.initial_edge_attr for _ in range(d.edge_index.shape[1])], dim=-2)
        d.h = torch.stack([self.initial_h for _ in range(d.num_nodes)], dim=-2)

    def description(self):
        return f"encoder: {torchinfo.summary(self.encoder_nn, verbose=0, depth=5)}\n" \
               f"processor (node hidden dim={self.hidden_dim}, edge attributes dim={self.edge_dim}):" \
               f" {torchinfo.summary(self.processor_nn, verbose=0, depth=5)}\n" \
               f"decoder: {torchinfo.summary(self.decoder_nn, verbose=0, depth=5)}"

    def save_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        encoder_path, decoder_path, processor_path, initial_hidden_path, initial_edge_attr_path = self.get_weights_paths(
            directory)
        for module, path in zip(
                [self.processor_nn, self.encoder_nn, self.decoder_nn],
                [processor_path, encoder_path, decoder_path]):
            torch.save(module.state_dict(), path)
        torch.save(self.initial_h, initial_hidden_path)
        torch.save(self.initial_edge_attr, initial_edge_attr_path)

