import torch
import torch.nn as nn
import torch_geometric as torch_g
import torch_scatter


class MaxMessagePassing(torch_g.nn.MessagePassing):
    def __init__(self, message_nn, final_nn):
        super(MaxMessagePassing, self).__init__(aggr="max", flow="target_to_source")
        self.message_nn = message_nn
        self.final_nn = final_nn

    def message(self, x_i, x_j, w):
        return self.message_nn(torch.cat([x_j, x_i, w], dim=-1))

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index=edge_index, x=x, w=edge_weight)

    def update(self, aggr_out, x):
        return self.final_nn(torch.cat([x, aggr_out], dim=-1))


# See "Benchmarking Graph Neural Networks" Dwivedi, Joshi, Laurent, Bengio and Bresson, https://arxiv.org/pdf/2003.00982.pdf page 22
# and "Graph Attention Networks" Velickovic, Cucurull, Casanova, Romero, Lio, Bengio, https://arxiv.org/pdf/1710.10903.pdf
class GatedGraphConvNetNoBatchNorm(torch.nn.Module):
    def __init__(self, node_dim):
        super(GatedGraphConvNetNoBatchNorm, self).__init__()
        edge_dim = node_dim
        self.U = torch.nn.Linear(node_dim, node_dim)
        self.V = torch.nn.Linear(node_dim, node_dim)
        self.A = torch.nn.Linear(node_dim, edge_dim)
        self.B = torch.nn.Linear(node_dim, edge_dim)
        self.C = torch.nn.Linear(edge_dim, edge_dim)

    def forward(self, x, edge_index, e_hat, batch_vector=None, eps=1e-12):
        x_i = x[edge_index[0], ...]
        x_j = x[edge_index[1], ...]

        e = torch_scatter.scatter_softmax(e_hat, edge_index[0], dim=0)
        y = self.V(x_j) * e
        y = torch_scatter.scatter_add(y, edge_index[0], dim=0, dim_size=x.shape[0])
        x = x + torch.nn.LeakyReLU(0.2)(self.U(x) + y)

        e_hat = e_hat + torch.nn.LeakyReLU(0.2)(self.A(x_i) + self.B(x_j) + self.C(e_hat))

        return x, e_hat


class MultilayerMPNN(torch.nn.Module):
    def construct_message_nn(self):
        return nn.Sequential(nn.Linear(2 * self.in_dim + self.edges_dim, self.message_dim), nn.ReLU())

    def construct_main_nn(self):
        return nn.Sequential(nn.Linear(self.in_dim + self.message_dim, self.in_dim), nn.ReLU())

    def __init__(self, in_dim, out_dim, message_dim, edges_dim, nr_hidden_layers=0):
        super(MultilayerMPNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.message_dim = message_dim
        self.edges_dim = edges_dim

        self.nr_hidden_layers = nr_hidden_layers
        self.layers = torch.nn.ModuleList()

        for layer_index in range(self.nr_hidden_layers):
            message_nn = self.construct_message_nn()
            main_nn = self.construct_main_nn()
            self.layers.append(MaxMessagePassing(message_nn=message_nn, final_nn=main_nn))

        message_nn = nn.Sequential(nn.Linear(2 * in_dim + edges_dim, message_dim), nn.ReLU())
        main_nn = nn.Sequential(nn.Linear(in_dim + message_dim, out_dim), nn.ReLU())
        self.layers += [MaxMessagePassing(message_nn=message_nn, final_nn=main_nn)]

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x


class ResidualMultilayerMPNN(MultilayerMPNN):
    def __init__(self, dim, message_dim, edges_dim, nr_layers=3):
        assert nr_layers >= 1
        super(ResidualMultilayerMPNN, self).__init__(dim, dim, message_dim, edges_dim, nr_hidden_layers=nr_layers - 1)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            x = x + layer(x, edge_index, edge_weight)
        return x


class MultilayerGatedGCN(torch.nn.Module):
    def __init__(self, dim, nr_layers=3):
        super(MultilayerGatedGCN, self).__init__()
        self.dim = dim
        self.nr_layers = nr_layers
        self.layers = nn.ModuleList()
        for layer_index in range(nr_layers):
            self.layers.append(GatedGraphConvNetNoBatchNorm(dim))

    def forward(self, x, edge_index, e_hat):
        for l in self.layers:
            x, e_hat = l(x, edge_index, e_hat)
        return x, e_hat

# Adding layer normalization

class ResidualMultilayerMPNNLayerNorm(ResidualMultilayerMPNN):
    def construct_message_nn(self):
        net = super().construct_message_nn()
        net.append(torch_g.nn.norm.LayerNorm(self.message_dim))
        return net

    def construct_main_nn(self):
        net = super().construct_main_nn()
        net.append(torch_g.nn.norm.LayerNorm(self.in_dim))
        return net
