import itertools
from os import walk
import os.path
from abc import ABC, abstractmethod
from typing import List
import numpy

import torch
from torch._C import Value
import torch.nn.functional as F
import torch_scatter
import torch_geometric as torch_g
import torchinfo

from src.NN_modules import ResidualMultilayerMPNN, MultilayerGatedGCN
from src.constants import MODEL_WEIGHTS_FOLDER


class WalkUpdater:
    @staticmethod
    def batch_prepare_for_first_step(d: torch_g.data.Batch, starting_nodes):
        d.x = torch.zeros([d.num_nodes, 3], device=d.edge_index.device)
        if starting_nodes is not None:
            d.x[starting_nodes, ...] = 1
        return d

    @staticmethod
    def update_state(d: torch_g.data.Data, current_nodes):
        previous = torch.squeeze(torch.nonzero(torch.eq(d.x[..., 1], torch.ones_like(d.x[..., 1]))), -1)

        if torch.any(current_nodes < 0).item() or torch.any(current_nodes > d.num_nodes).item():
            print("Illegal choice of next step when updating graph mask")
            raise Exception("Illegal choice of next step node when updating graph mask!")

        d.x[previous, 1] = 0
        d.x[current_nodes, 1] = 1
        d.x[current_nodes, 2] = 1


class HamFinder(ABC):
    @abstractmethod
    def solve_graphs(self, graphs: List[torch_g.data.Data]) -> List[int]:
        pass


class HamFinderGNN(HamFinder):
    def __init__(self, graph_updater: WalkUpdater):
        self.graph_updater = graph_updater

    @abstractmethod
    def next_step_logits(self, d: torch_g.data.Batch) -> torch.Tensor:
        pass

    def next_step_prob(self, d: torch_g.data.Batch) -> torch.Tensor:
        logits = self.next_step_logits(d)
        p = torch_scatter.scatter_softmax(logits, d.batch)
        return p

    @staticmethod
    def _neighbor_mask(d: torch_g.data.Data):
        current = torch.nonzero(torch.isclose(d.x[..., 1], torch.ones_like(d.x[..., 1]))).squeeze(-1)
        if current.numel == 0:
            return torch.ones_like(d.x[..., 1])
        neighbor_index = d.edge_index[1, torch.any(d.edge_index[None, 0, :] == current[:, None], dim=0)]
        return neighbor_index.unique()

    @staticmethod
    def _mask_neighbor_logits(logits, d: torch_g.data.Data):
        valid_next_step_indices = torch.nonzero(HamFinderGNN._neighbor_mask(d)).squeeze(-1)
        neighbor_logits = torch.zeros_like(logits).log()
        neighbor_logits[valid_next_step_indices] = logits.index_select(0, valid_next_step_indices)
        return neighbor_logits

    def next_step_logits_masked_over_neighbors(self, d: torch_g.data.Batch) -> torch.Tensor:
        if torch.all(torch.lt(d.x[..., 1], 0.5)):
            return self.next_step_logits(d)
        logits = self.next_step_logits(d)
        return self._mask_neighbor_logits(logits, d)

    def next_step_prob_masked_over_neighbors(self, d: torch_g.data.Batch) -> torch.Tensor:
        logits = self.next_step_logits_masked_over_neighbors(d)
        neighbor_prob = torch_scatter.scatter_softmax(logits, d.batch)
        return neighbor_prob

    def prepare_for_first_step(self, d: torch_g.data.Batch, start_batch):
        return self.graph_updater.batch_prepare_for_first_step(d, start_batch)

    def update_state(self, d: torch_g.data.Data, current_batch):
        return self.graph_updater.update_state(d, current_batch)

    @abstractmethod
    def init_graph(self, d) -> None:
        pass

    @abstractmethod
    def to(self, device) -> None:
        pass

    @abstractmethod
    def get_device(self) -> str:
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def load_weights(self, out_folder):
        pass

    @abstractmethod
    def save_weights(self, out_folder):
        pass

    def _neighbor_prob_and_greedy_choice_for_batch(self, d: torch_g.data.Batch):
        p = self.next_step_prob_masked_over_neighbors(d)
        graph_sizes = [g.num_nodes for g in d.to_data_list()]
        max_size = max(graph_sizes)
        p = torch.stack([F.pad(x, (0, max_size - x.shape[0]), "constant", -1) for x in torch.split(p, graph_sizes)])

        choice = torch.argmax(
            torch.isclose(p, torch.max(p, dim=-1)[0][..., None])
            * (p + torch.randperm(p.shape[-1], device=p.device)[None, ...]), dim=-1)
        choice += torch.tensor(list(itertools.accumulate(graph_sizes[:-1], initial=0)))

        return p, choice

    def batch_run_greedy_neighbor(self, batch_data: torch_g.data.Batch):
        with torch.no_grad():
            nodes_per_graph = [d.num_nodes for d in batch_data.to_data_list()]
            max_steps_in_a_cycle = max(nodes_per_graph) + 1

            self.init_graph(batch_data)
            all_choices, all_scores = [], []
            choice = None
            stop_algorithm_mask = torch.zeros([batch_data.num_graphs], device=batch_data.edge_index.device, dtype=torch.uint8)

            for step in range(0, max_steps_in_a_cycle):
                if step < 2:
                    self.prepare_for_first_step(batch_data, choice)
                else:
                    current_nodes = choice
                    self.update_state(batch_data, current_nodes[current_nodes != -1])
                p, choice = self._neighbor_prob_and_greedy_choice_for_batch(batch_data)
                choice = torch.logical_not(stop_algorithm_mask) * choice - stop_algorithm_mask
                p = p * torch.logical_not(stop_algorithm_mask)[..., None]- stop_algorithm_mask[..., None]
                if len(all_choices) > 0:
                    stop_algorithm_mask = torch.maximum(
                        stop_algorithm_mask, torch.any(torch.stack([torch.eq(x, choice) for x in all_choices])))
                all_choices.append(choice)
                all_scores.append(p)

                if torch.all(stop_algorithm_mask).item():
                    break
            walks = torch.stack(all_choices, -1)
            selections = torch.stack(all_scores, -2)
            if walks.shape[-1] != max_steps_in_a_cycle:
                walks = F.pad(walks, (0, max_steps_in_a_cycle - walks.shape[-1]), value=-1)
            if selections.shape[-2] != max_steps_in_a_cycle: 
                selections = F.pad(selections, (0, 0, 0, max_steps_in_a_cycle - selections.shape[-2]), value=-1)
            return walks, selections

    def get_batch_size_for_multi_solving(self):
        return 8

    def solve_graphs(self, graph_generator):
        batch_size = self.get_batch_size_for_multi_solving()
        batch_generator = ([first] + [d for d in itertools.islice(graph_generator, batch_size - 1)]
                           for first in graph_generator)

        walks = []
        for list_of_graphs in batch_generator:
            batch_shift = numpy.cumsum([0] + [g.num_nodes for g in list_of_graphs[:-1]])
            batch_graph = torch_g.data.Batch.from_data_list(list_of_graphs)
            walks_tensor, _ = self.batch_run_greedy_neighbor(batch_graph)
            walks_tensor -= batch_shift[:, None]
            walks_tensor[walks_tensor < 0] = -1
            raw_walks = [[int(node.item()) for node in walk] for walk in walks_tensor]
            for walk in raw_walks:
                walk_end_index = walk.index(-1) if -1 in walk else len(walk)
                walks.append(walk[:walk_end_index])
        return walks


class HamCycleFinderWithValueFunction(HamFinderGNN):
    @abstractmethod
    def next_step_logits_and_value_function(self, d:torch_g.data.Data) -> [torch.Tensor, torch.Tensor]:
        pass


class EncodeProcessDecodeAlgorithm(HamFinderGNN):
    def _construct_processor(self):
        return ResidualMultilayerMPNN(dim=self.hidden_dim, message_dim=self.hidden_dim, edges_dim=1,
                                      nr_layers=self.processor_depth)

    def _construct_encoder_and_decoder(self):
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(3 + self.hidden_dim, self.hidden_dim))
        decoder_nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + self.hidden_dim, self.out_dim))
        return encoder_nn, decoder_nn

    def __init__(self, is_load_weights=True, processor_depth=3, in_dim=1, out_dim=1, hidden_dim=32,
                 device="cpu", graph_updater=WalkUpdater()):
        super(EncodeProcessDecodeAlgorithm, self).__init__(graph_updater)
        self.PROCESSOR_NAME = f"{self.__class__.__name__}_Processor.tar"
        self.ENCODER_NAME = f"{self.__class__.__name__}_Encoder.tar"
        self.DECODER_NAME = f"{self.__class__.__name__}_Decoder.tar"
        self.INITIAL_HIDDEN_TENSOR_NAME = f"{self.__class__.__name__}_InitialHiddenTensor.pt"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.processor_depth = processor_depth

        self.encoder_nn, self.decoder_nn = self._construct_encoder_and_decoder()
        self.processor_nn = self._construct_processor()
        self.initial_h = torch.rand(self.hidden_dim, device=self.device)
        self.to(device)

        if is_load_weights:
            self.load_weights()

    def to(self, device):
        for module in [self.decoder_nn, self.encoder_nn, self.processor_nn]:
            module.to(device)
        self.initial_h = self.initial_h.to(device)
        self.device = device

    def to_cuda(self):
        self.to("cuda")

    def get_device(self):
        return self.device

    def description(self):
        return f"encoder: {torchinfo.summary(self.encoder_nn, verbose=0, depth=5)}\n" \
               f"processor (hidden dim={self.hidden_dim}):" \
               f" {torchinfo.summary(self.processor_nn, verbose=0, depth=5)}\n" \
               f"decoder: {torchinfo.summary(self.decoder_nn, verbose=0, depth=5)}"

    def get_weights_paths(self, directory):
        return [os.path.join(directory, name) for name in
                [self.ENCODER_NAME, self.DECODER_NAME, self.PROCESSOR_NAME, self.INITIAL_HIDDEN_TENSOR_NAME]]

    def load_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        encoder_path, decoder_path, processor_path, initial_hidden_path = self.get_weights_paths(directory)
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

    def parameters(self):
        return itertools.chain(self.encoder_nn.parameters(), self.decoder_nn.parameters(),
                               self.processor_nn.parameters(), [self.initial_h])

    def next_step_logits(self, d: torch_g.data.Data):
        d.z = self.encoder_nn(torch.cat([d.x, d.h], dim=-1))
        d.h = self.processor_nn(d.z, d.edge_index, d.edge_attr)
        return torch.squeeze(self.decoder_nn(torch.cat([d.z, d.h], dim=-1)), dim=-1)

    def init_graph(self, d):
        #TODO edge features are useless here but are needed because of how layers are implemented at the moment
        d.edge_attr = torch.ones([d.edge_index.shape[1], 1], device=self.device)
        d.h = torch.stack([self.initial_h for _ in range(d.num_nodes)], dim=-2)

    def save_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        encoder_path, decoder_path, processor_path, initial_hidden_path = self.get_weights_paths(directory)
        for module, path in zip(
                [self.processor_nn, self.encoder_nn, self.decoder_nn],
                [processor_path, encoder_path, decoder_path]):
            torch.save(module.state_dict(), path)
        torch.save(self.initial_h, initial_hidden_path)


class EmbeddingAndMaxMPNN(HamCycleFinderWithValueFunction):
    def _construct_embedding(self):
        embedding = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.embedding_depth)
        embedding_out_projection = torch.nn.Linear(self.hidden_dim, self.hidden_dim - 3)
        return embedding, embedding_out_projection

    def _construct_processor(self):
        processor = ResidualMultilayerMPNN(self.hidden_dim, self.hidden_dim, 1, nr_layers=self.processor_depth)
        processor_out_projection = torch.nn.Linear(self.hidden_dim, self.out_dim)
        return processor, processor_out_projection

    def __init__(self, is_load_weights=True, in_dim=3, out_dim=2, hidden_dim=32, embedding_depth=5, processor_depth=5,
                 device="cpu", graph_updater=WalkUpdater()):
        super(EmbeddingAndMaxMPNN, self).__init__(graph_updater)
        self.EMBEDDING_NAME = "{}-Embedding.tar".format(self.__class__.__name__)
        self.PROCESSOR_NAME = "{}-Processor.tar".format(self.__class__.__name__)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embedding_depth = embedding_depth
        self.processor_depth = processor_depth
        assert hidden_dim > 3
        self.hidden_dim = hidden_dim
        self.device = device
        self.softmax = torch.nn.Softmax(dim=0)
        self.initial_embedding = torch.rand([hidden_dim], device=self.device, requires_grad=True)
        self.embedding, self.embedding_out_projection = self._construct_embedding()
        self.processor, self.processor_out_projection = self._construct_processor()

        if device == "cuda":
            self.to_cuda()

        if is_load_weights:
            self.load_weights()

    def to(self, device):
        for module in [self.embedding, self.embedding_out_projection, self.processor, self.processor_out_projection]:
            module.to(device)
        self.initial_embedding = self.initial_embedding.to(device)
        self.device = device

    def to_cuda(self):
        self.initial_embedding.to("cuda")
        for module in [self.embedding, self.embedding_out_projection, self.processor, self.processor_out_projection]:
            module.cuda()
        self.device = "cuda"

    def embed(self, d: torch_g.data.Data()):
        d.emb = self.embedding_out_projection(
            self.embedding.forward(torch.unsqueeze(self.initial_embedding, 0).expand(d.num_nodes, -1),
                                   d.edge_index, d.edge_attr))

    def forward(self, d: torch_g.data.Data) -> torch.Tensor:
        return self.processor_out_projection(
            torch.cat([d.emb, d.x], dim=1) + self.processor(torch.cat([d.emb, d.x], dim=1), d.edge_index, d.edge_attr))

    def next_step_logits(self, d: torch_g.data.Data) -> torch.Tensor:
        return self.forward(d)[..., 0]

    def next_step_logits_and_value_function(self, d: torch_g.data.Data) -> [torch.Tensor, torch.Tensor]:
        result = self.forward(d)
        logits = result[..., 0]
        value_estimate = torch.max(result[..., 1], -1)[0]
        if d.x[..., 0].any():
            return HamFinderGNN._mask_neighbor_logits(logits, d), value_estimate
        else:
            return logits, value_estimate

    def init_graph(self, d) -> None:
        d.edge_attr = torch.ones([d.edge_index.shape[1], 1], device=self.device)
        self.embed(d)

    def get_device(self) -> str:
        return self.device

    def parameters(self):
        return itertools.chain([self.initial_embedding], self.embedding.parameters(),
                               self.embedding_out_projection.parameters(), self.processor.parameters(),
                               self.processor_out_projection.parameters())

    def description(self):
        return f"embedding (hidden dim={self.hidden_dim}): {torchinfo.summary(self.embedding, verbose=0, depth=5)}\n" \
               f"processor (hidden dim={self.hidden_dim}): {torchinfo.summary(self.processor, verbose=0, depth=5)}"

    def get_weights_paths(self, directory):
        return [os.path.join(directory, name) for name in [self.EMBEDDING_NAME, self.PROCESSOR_NAME]]

    def load_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        embedding_path, processor_path = self.get_weights_paths(directory)
        if os.path.isfile(embedding_path):
            embedding_save_data = torch.load(embedding_path)
            self.initial_embedding = embedding_save_data["initial"]["initial"]
            for module_key, module in zip(["MPNN", "out_projection"], [self.embedding, self.embedding_out_projection]):
                module.load_state_dict(embedding_save_data[module_key])
            print("Loaded embedding module from {}".format(embedding_path))
        if os.path.isfile(processor_path):
            processor_save_data = torch.load(processor_path)
            for modul_key, module in zip(["MPNN", "out_projection"], [self.processor, self.processor_out_projection]):
                module.load_state_dict(processor_save_data[modul_key])
            print("Loaded processor module from {}".format(processor_path))

    def save_weights(self, directory=MODEL_WEIGHTS_FOLDER):
        embedding_path, processor_path = self.get_weights_paths(directory)
        embedding_save_data = {
            'initial': {"initial": self.initial_embedding},
            'MPNN': self.embedding.state_dict(),
            'out_projection': self.embedding_out_projection.state_dict()
        }
        processor_save_data = {
            'MPNN': self.processor.state_dict(),
            'out_projection': self.processor_out_projection.state_dict()
        }
        for save_data, path in zip([embedding_save_data, processor_save_data],
                                   [embedding_path, processor_path]):
            torch.save(save_data, path)


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
        d.edge_attr = torch.ones([d.edge_index.shape[1], self.hidden_dim], device=self.device)
        self.embed(d)
