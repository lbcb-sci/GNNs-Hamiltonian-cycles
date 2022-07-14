
import torch
import torch_geometric as torch_g

from src.nn_modules.nn_modules import MultilayerGatedGCN
from src.nn_modules.EmbeddingAndMaxMPNN import EmbeddingAndMaxMPNN


class GatedGCNEmbedAndProcess(EmbeddingAndMaxMPNN):
    def _construct_embedding(self):
        embedding = MultilayerGatedGCN(self.hidden_dim, self.embedding_depth)
        embedding_out_projection = torch.nn.Linear(self.hidden_dim, self.hidden_dim - 3)
        return embedding, embedding_out_projection

    def _construct_processor(self):
        processor = MultilayerGatedGCN(self.hidden_dim, self.processor_depth)
        processor_out_projection = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return processor, processor_out_projection

    def embed(self, d: torch_g.data.Data()):
        emb, d.edge_attr = self.embedding.forward(torch.squeeze(self.initial_embedding, 0).expand(d.num_nodes, -1),
                                                  d.edge_index, d.edge_attr.squeeze(-1))
        d.emb = self.embedding_out_projection(emb)

    def forward(self, d: torch_g.data.Data) -> torch.Tensor:
        x, e_hat = self.processor(torch.cat([d.emb, d.x], dim=1), d.edge_index, d.edge_attr)
        return self.processor_out_projection(x)

    def init_graph(self, d) -> None:
        d.edge_attr = torch.ones([d.edge_index.shape[1], self.hidden_dim]) # TODO this is duplicated from superclass (also, would be nice to only do it once and pass embeding through dataloaders)
        self.embed(d)
