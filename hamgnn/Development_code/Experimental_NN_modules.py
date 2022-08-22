import torch
import torch.nn as nn
import torch_scatter

from src.NN_modules import ResidualMultilayerMPNN


class ResidualMultilayerMPNN3DeepMessages(ResidualMultilayerMPNN):
    def construct_message_nn(self):
        return nn.Sequential(
            nn.Linear(2 * self.in_dim + self.edges_dim, self.message_dim),
            nn.ReLU(),
            nn.Linear(self.message_dim, self.message_dim),
            nn.ReLU(),
            nn.Linear(self.message_dim, self.out_dim)
        )


class AttentionMPNNWithEdgeFeatures(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, dropout_rate=0.1):
        super(AttentionMPNNWithEdgeFeatures, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.dropout_rate = dropout_rate
        self.message_nn = torch.nn.Sequential(
            torch.nn.Linear(node_dim*2 + edge_dim, node_dim),
            torch.nn.ReLU(), torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(), torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(node_dim, node_dim))
        self.edge_update_nn = torch.nn.Sequential(
            torch.nn.Linear(node_dim*2 + edge_dim, edge_dim),
            torch.nn.ReLU(), torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(edge_dim, edge_dim),
            torch.nn.ReLU(), torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(edge_dim, edge_dim))
        self.attention_nn = torch.nn.Linear(node_dim*2 + edge_dim, 1)

    def forward(self, node_attr, edge_attr, edge_index):
        x = node_attr
        e = edge_attr
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        messages = self.message_nn.forward(torch.cat([x_i, x_j, e], dim=-1))
        attention = torch.squeeze(self.attention_nn.forward(torch.cat([x_i, x_j, e], dim=-1)), dim=-1)
        attention = torch_scatter.scatter_softmax(attention, edge_index[0])
        tmp = attention[..., None]*messages
        x = torch_scatter.scatter_sum(attention[..., None]*messages, edge_index[0], dim=-2, dim_size=node_attr.shape[0])
        e = self.edge_update_nn(torch.cat([x_i, x_j, e], dim=-1))
        return x, e


class ResidualMultilayerAttentionMPNNWithEdgeFeatures(torch.nn.Module):
    def __init__(self, node_dimension, edge_dimension, nr_layers, dropout_rate=0.1):
        super(ResidualMultilayerAttentionMPNNWithEdgeFeatures, self).__init__()
        self.node_dimension = node_dimension
        self.edge_dimension = edge_dimension
        self.nr_layers = nr_layers
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        for _ in range(self.nr_layers):
            self.layers.append(AttentionMPNNWithEdgeFeatures(self.node_dimension, self.edge_dimension, self.dropout_rate))

    def forward(self, node_attr, edge_attr, edge_index):
        x, e = node_attr, edge_attr
        for layer in self.layers:
            x_update, e_update = layer(x, e, edge_index)
            x += x_update
            e += e_update
        return x, e
