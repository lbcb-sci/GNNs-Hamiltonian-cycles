from abc import ABC, abstractmethod
from typing import Dict
import itertools
import torch
import pandas as pd
import os
import torch.multiprocessing as mp
from portalocker import Lock
import time

from DBModelRecording import EVALUATION_DATABASE_PATH, create_archive_evaluation_entry
from src.Models import EmbeddingAndMaxMPNN, EncodeProcessDecodeAlgorithm, GatedGCNEmbedAndProcess
from src.Development_code.ExperimentalModels import GatedGCNEncodeProcessDecodeAlgorithm, EncodeProcessDecodeWithEdgeFeatures, EncodeProcessDecodeWithDeepMessagesAlgorithm
from src.Trainers import SupervisedTrainFollowingHamiltonCycle,\
    REINFORCE_WithLearnableBaseline, plot_history
from src.Development_code.ExperimentalTrainers import NeighborMaskedSupervisedTrainFollowingHamiltonCycle, REINFOCE_With_Averaged_simulations
from src.GraphGenerators import ErdosRenyiGenerator, NoisyCycleBatchGenerator
from src.constants import EVALUATION_DATABASE_LOCK_PATH


class HamiltonModelLogRequest(ABC):
    def __init__(self, short_description, learn_rate, train_epochs, iterations_in_epoch):
        self.short_description = short_description
        self.learn_rate = learn_rate
        self.train_epochs = train_epochs
        self.iterations_in_epoch = iterations_in_epoch

    def run_log_request(self, database_lock):
        model_entry = self._process_request()
        with database_lock:
            df = pd.read_csv(EVALUATION_DATABASE_PATH) if os.path.isfile(EVALUATION_DATABASE_PATH) \
                else pd.DataFrame()
            df = df.append(model_entry, ignore_index=True)
            with open(EVALUATION_DATABASE_PATH, "w", encoding="utf-8") as file:
                df.to_csv(file, encoding="utf-8", header=True, index=False)

    @abstractmethod
    def _process_request(self) -> Dict[str, str]: pass


class BatchEmbeddingProcessErdosRenyiReinforcedLogRequest(HamiltonModelLogRequest):
    def __init__(self, short_description, train_graph_size, train_graph_ham_prob, learn_rate, train_epochs, iterations_in_epoch,
                 hidden_dim, batch_size=1, embedding_depth=5, processor_depth=5, episode_per_example=1, loss_l2_regularization_weight=0.01,
                 loss_value_function_weight=1, save_train_history=False):
        super(BatchEmbeddingProcessErdosRenyiReinforcedLogRequest, self).__init__(short_description, learn_rate,
                                                                             train_epochs, iterations_in_epoch)
        self.train_graph_size = train_graph_size
        self.train_graph_ham_prob = train_graph_ham_prob
        self.hidden_dim = hidden_dim
        self.embedding_depth = embedding_depth
        self.processor_depth = processor_depth
        self.loss_l2_regularization_weight = loss_l2_regularization_weight
        self.loss_value_function_weight = loss_value_function_weight
        self.batch_size = batch_size
        self.episodes_per_example = episode_per_example
        self.save_train_history = save_train_history

    def _get_model(self):
        return EmbeddingAndMaxMPNN(False, hidden_dim=self.hidden_dim, embedding_depth=self.embedding_depth,
                                   processor_depth=self.processor_depth)

    def _get_trainer(self):
        return REINFORCE_WithLearnableBaseline(
            self.train_epochs, self.iterations_in_epoch, episodes_per_example=self.episodes_per_example,
            l2_regularization_weight=self.loss_l2_regularization_weight,
            value_function_weight=self.loss_value_function_weight, simulation_batch_size=self.batch_size)

    def _get_generator(self):
        return ErdosRenyiGenerator(self.train_graph_size, self.train_graph_ham_prob)

    def _process_request(self):
        self.reinforced_model = self._get_model()
        parameters = set(itertools.chain(self.reinforced_model.parameters()))
        self.optimizer = torch.optim.Adam(list(parameters), lr=self.learn_rate)
        self.trainer = self._get_trainer()
        self.generator = self._get_generator()
        history = self.trainer.train(self.generator, self.reinforced_model, self.optimizer)
        model_entry = create_archive_evaluation_entry(
            self.reinforced_model, self.generator, "Reinforcement Embedding-Process model trained on Erdos-Renyi graphs", history,
            self.iterations_in_epoch, self.learn_rate, self.optimizer, "Reinfocement_loss",
            self.trainer.loss_description(), description=self.short_description,
            training_graphs_seen=self.train_epochs, sample_method="greedy",
            train_description=self.trainer.train_description(), batch_size=self.batch_size)
        if self.save_train_history:
            plot_history(history, f"{self.short_description}_{time.time_ns()}_train.png", "Reinforcement loss:")
        return model_entry


class GatedGCNEmbedProcessERReinforcedRequest(BatchEmbeddingProcessErdosRenyiReinforcedLogRequest):
    def _get_model(self):
        return GatedGCNEmbedAndProcess(False, hidden_dim=self.hidden_dim, embedding_depth=self.embedding_depth,
                                       processor_depth=self.processor_depth)


class AveragingGatedEPReinforceRequest(GatedGCNEmbedProcessERReinforcedRequest):
    def _get_trainer(self):
        return REINFOCE_With_Averaged_simulations(
            self.train_epochs, self.iterations_in_epoch, self.loss_l2_regularization_weight,
            self.loss_value_function_weight, simulation_batch_size=self.batch_size)


class EmbeddingProcessSupervisedTrainRequest(HamiltonModelLogRequest):
    def __init__(self, short_description, train_graph_size,
                 learn_rate, train_epochs, iterations_in_epoch,
                 hidden_dim, batch_size=1, nr_precoded_cycles_in_train=1, nr_expected_noise_edges_per_node=1,
                 embedding_depth=5, processor_depth=5, loss_type="entropy"):
        super(EmbeddingProcessSupervisedTrainRequest, self).__init__(short_description, learn_rate, train_epochs, iterations_in_epoch)
        self.embedding_depth = embedding_depth
        self.processor_depth = processor_depth
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.train_graph_size = train_graph_size
        self.nr_precoded_cycles_in_train = nr_precoded_cycles_in_train
        self.nr_expected_noise_edges_per_node = nr_expected_noise_edges_per_node
        self.loss_type = loss_type

    def _process_request(self) -> Dict[str, str]:
        self.model = EmbeddingAndMaxMPNN(False, self.embedding_depth, self.processor_depth)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learn_rate)
        self.trainer = SupervisedTrainFollowingHamiltonCycle(self.train_epochs, self.iterations_in_epoch,
                                                             self.loss_type, batch_size=self.batch_size)
        self.generator = NoisyCycleBatchGenerator(self.train_graph_size,
                                        self.nr_expected_noise_edges_per_node, batch_size=self.batch_size)
        history = self.trainer.train(self.generator, self.model, self.optimizer)
        model_entry = create_archive_evaluation_entry(
            self.model, self.generator, "Supervised Embed-Process model trained on noisy cycles", history,
            self.iterations_in_epoch, self.learn_rate, self.optimizer, f"{self.loss_type} supervised loss",
            self.trainer.loss_description(), description=self.short_description,
            training_graphs_seen=self.train_epochs * self.iterations_in_epoch, sample_method="greedy",
            train_description=self.trainer.train_description(), batch_size=self.batch_size)
        return model_entry


class BatchEncodeProcessDecodeSupervisedTrainRequest(HamiltonModelLogRequest):
    def __init__(self, short_description, train_graph_size,
                 learn_rate, train_epochs, iterations_in_epoch,
                 hidden_dim, batch_size=1, nr_precoded_cycles_in_train=1, nr_expected_noise_edges_per_node=1,
                 processor_depth=5, loss_type="entropy", save_train_history=False):
        super(BatchEncodeProcessDecodeSupervisedTrainRequest, self).__init__(short_description, learn_rate, train_epochs,
                                                                        iterations_in_epoch)
        self.short_description = short_description
        self.train_graph_size = train_graph_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nr_precoded_cycles_in_train = nr_precoded_cycles_in_train
        self.nr_expected_noise_edges_per_node = nr_expected_noise_edges_per_node
        self.processor_depth = processor_depth
        self.loss_type = loss_type
        self.save_train_history = save_train_history

    def _get_trainer(self):
        return SupervisedTrainFollowingHamiltonCycle(self.train_epochs, self.iterations_in_epoch,
                                                     self.loss_type, batch_size=self.batch_size)

    def _get_hamilton_model(self):
        return EncodeProcessDecodeAlgorithm(False, processor_depth=self.processor_depth, hidden_dim=self.hidden_dim)

    def _get_generator(self):
        return NoisyCycleBatchGenerator(self.train_graph_size,
                                        self.nr_expected_noise_edges_per_node, self.batch_size)

    def _process_request(self):
        self.supervised_model = self._get_hamilton_model()
        parameters = set(itertools.chain(self.supervised_model.parameters()))
        self.optimizer = torch.optim.Adam(list(parameters), lr=self.learn_rate)
        self.trainer = self._get_trainer()
        generator = self._get_generator()
        history = self.trainer.train(generator, self.supervised_model, self.optimizer)
        model_entry = create_archive_evaluation_entry(
            self.supervised_model, generator, "Supervised Encode-Process-Decode model trained on noisy cycles", history,
            self.iterations_in_epoch, self.learn_rate, self.optimizer, f"{self.loss_type} supervised loss",
            self.trainer.loss_description(), description=self.short_description,
            training_graphs_seen=self.train_epochs*self.iterations_in_epoch, sample_method="greedy",
            train_description=self.trainer.train_description(), batch_size=self.batch_size)
        if self.save_train_history:
            plot_history(history, f"{self.short_description}_{time.time_ns()}_train.png")

        return model_entry


class EncodeProcessDecodeSupervisedMaskNeighborsRequest(BatchEncodeProcessDecodeSupervisedTrainRequest):
    def _get_trainer(self):
        return NeighborMaskedSupervisedTrainFollowingHamiltonCycle(self.train_epochs, self.iterations_in_epoch,
                                              self.loss_type, verbose=1)


class EncodeProcessDecodeSupervisedGatedGCNRequest(BatchEncodeProcessDecodeSupervisedTrainRequest):
    def _get_hamilton_model(self):
        return GatedGCNEncodeProcessDecodeAlgorithm(False, processor_depth=self.processor_depth)


class EncodeProcessDecodeSupervisedAttentionWithEdgeFeaturesRequest(BatchEncodeProcessDecodeSupervisedTrainRequest):
    def _get_hamilton_model(self):
        return EncodeProcessDecodeWithEdgeFeatures(False, processor_depth=2)


class BatchEncodeProcessDecodeSupervisedDeepMessagesRequest(BatchEncodeProcessDecodeSupervisedTrainRequest):
    def _get_hamilton_model(self):
        return EncodeProcessDecodeWithDeepMessagesAlgorithm(False, processor_depth=self.processor_depth,
                                                            hidden_dim=self.hidden_dim)


class DecayingLRBatchEPDSupervisedLogRequest(BatchEncodeProcessDecodeSupervisedTrainRequest):
    def _get_trainer(self):
        return SupervisedTrainFollowingHamiltonCycle(self.train_epochs, self.iterations_in_epoch,
                                                     self.loss_type, batch_size=self.batch_size,
                                                     learning_rate_decrease_condition_window=50,
                                                     terminating_learning_rate=1e-10)


def log_request_starter(log_request: HamiltonModelLogRequest):
    log_request.run_log_request(lock)
    return "OK"


def pool_initializer(l):
    global lock
    lock = l


if __name__ == '__main__':
    mp.set_start_method("spawn")
    torch.autograd.set_detect_anomaly(True)

    NR_THREADS = 4
    torch.set_num_threads(NR_THREADS)
    print(f"Working on max {NR_THREADS} parallel threads")

    if not os.path.exists(EVALUATION_DATABASE_LOCK_PATH):
        with open(EVALUATION_DATABASE_LOCK_PATH, "w") as lock_file:
            lock_file.writelines("")
    database_lock = Lock(EVALUATION_DATABASE_LOCK_PATH)

    layer_depth_reinforcement_requests = [BatchEmbeddingProcessErdosRenyiReinforcedLogRequest(
        "Embed-Process model trained with reinforcement for finding the optimal depth",
        train_graph_size=10, train_graph_ham_prob=0.8, learn_rate=0.0001, train_epochs=100, iterations_in_epoch=100,
        hidden_dim=32, embedding_depth=depth, processor_depth=depth) for depth in [1, 2, 3, 4, 5, 7, 10]]

    hamilton_existance_prob_reinforcement_requests = [BatchEmbeddingProcessErdosRenyiReinforcedLogRequest(
        "Embed-Process model trained with reinforc for finding the optimal probability of Hamilton cycle existance in train graphs",
        train_graph_size=10, train_graph_ham_prob=p, learn_rate=0.0001, train_epochs=100, iterations_in_epoch=100,
        hidden_dim=32) for p in [0.2, 0.5, 0.7, 0.8, 0.9]]

    layer_depth_long_train_requests = [BatchEmbeddingProcessErdosRenyiReinforcedLogRequest(
        "Deep Embed-Process trained for longer to find convergence time",
        train_graph_size=10, train_graph_ham_prob=0.8, learn_rate=lr, train_epochs=epochs, iterations_in_epoch=200,
        hidden_dim=32, embedding_depth=depth, processor_depth=depth) for depth in [7, 10]
        for epochs in [200, 500, 1000] for lr in [1e-3, 1e-4, 1e-5]]

    deep_embed_shallow_process_request = [BatchEmbeddingProcessErdosRenyiReinforcedLogRequest(
        "Moving all the complexity to embedding", 10, train_graph_ham_prob=0.8, learn_rate=1e-4, train_epochs=600,
        iterations_in_epoch=200, hidden_dim=32, embedding_depth=depth, processor_depth=1)
        for depth in [5, 10, 15, 20]]

    supervised_size_learn_rate_and_train_time = [BatchEncodeProcessDecodeSupervisedTrainRequest(
        "Encode-process-decode on different train sizes, learn rates, and train length", size, lr, epochs,
        iterations_in_epoch=100, hidden_dim=32)
        for size in [10, 20, 35, 50, 70] for lr in [1e-4, 1e-5] for epochs in [100, 500, 1000]]

    supervised_size_particular_choices = [BatchEncodeProcessDecodeSupervisedTrainRequest(
        "Encode-process-decode on different train sizes, learn rates, and train length", size, lr, epochs,
        iterations_in_epoch=100, hidden_dim=32)
        for size in [20, 25, 30, 35] for lr in [1e-4, 5*1e-5] for epochs in [2000, 3000]]

    high_quality_supervised_embed_process_requests = \
        [EmbeddingProcessSupervisedTrainRequest(
            f"Conv. speed for supervised embed-process on {epochs} epochs", 25, 1e-4, train_epochs=epochs,
            iterations_in_epoch=100, hidden_dim=32)
            for epochs in [200, 500, 1000, 1500, 2000]] + \
        [EmbeddingProcessSupervisedTrainRequest(
            f"Processor {depth} layers deep in supervised embed-process", 25, 1e-4, train_epochs=500,
            iterations_in_epoch=100, hidden_dim=32, processor_depth=depth)
            for depth in [2, 3, 5, 7, 10, 15]] + \
        [EmbeddingProcessSupervisedTrainRequest(
            f"Hidden dimension of {hidden} in supervised embed-process", 25, 1e-4, train_epochs=500,
            iterations_in_epoch=100, hidden_dim=hidden)
            for hidden in [8, 16, 32, 64, 128, 256]]

    neighbor_mask_for_supervised_training_on_large_sets_request = [EncodeProcessDecodeSupervisedMaskNeighborsRequest(
        f"Neighbors masked for training on {size}-graphs, supervised encode-process-decode", size, 1e-4, train_epochs=epochs,
        iterations_in_epoch=100, hidden_dim=32)
        for size in [30, 50, 70, 100] for epochs in [500, 1000]]

    supervised_embed_process_request = [EmbeddingProcessSupervisedTrainRequest(
        f"Supervised Embed-Process on noisy {train_size}-cycles over {epochs} epochs", train_size, 1e-4, train_epochs=epochs,
        iterations_in_epoch=100, hidden_dim=32)
        for train_size in [10, 20, 30, 70] for epochs in [500, 1000]]

    vary_train_size_reinforce_request = [BatchEmbeddingProcessErdosRenyiReinforcedLogRequest(
        f"Shallow Embed-Process net trained on ER({train_size}, {0.8}) for {epochs}",
        train_graph_size=train_size, train_graph_ham_prob=0.8, learn_rate=1e-4, train_epochs=epochs,
        iterations_in_epoch=200, hidden_dim=32, embedding_depth=2, processor_depth=2)
        for train_size in [10, 20, 30, 50, 70] for epochs in [200, 500]]

    high_quality_supervised_encode_process_decode_requests = \
        [BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"Conv. speed for supervised encode-process-decode on {epochs} epochs", 25, 1e-4, train_epochs=epochs,
            iterations_in_epoch=100, hidden_dim=32)
            for epochs in [200, 500, 1000, 1500, 2000]] + \
        [BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"Processor {depth} layers deep in supervised encode-process-decode", 25, 1e-4, train_epochs=500,
            iterations_in_epoch=100, hidden_dim=32, processor_depth=depth)
            for depth in [2, 3, 5, 7, 10, 15]] + \
        [BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"Hidden dimension of {hidden} in supervised encode-process-decode", 25, 1e-4, train_epochs=500,
            iterations_in_epoch=100, hidden_dim=hidden)
            for hidden in [8, 16, 32, 64, 128, 256]]

    hq_neighbor_masked_encode_process_decode_requests = [EncodeProcessDecodeSupervisedMaskNeighborsRequest(
        f"Hq neighbors masked on {size}-graphs, supervised encode-process-decode", size, 1e-4, train_epochs=epochs,
        iterations_in_epoch=100, hidden_dim=32)
        for size in [25, 30, 50] for epochs in [2000]]

    __train_epochs = 4000
    long_train_best_model_requests = [BatchEncodeProcessDecodeSupervisedTrainRequest(
        f"Long train of currently best e-p-d model, {__train_epochs} epochs", 25, 1e-4, train_epochs=__train_epochs,
        iterations_in_epoch=100, hidden_dim=32)]

    gated_GCN_supervised_various_requests = [EncodeProcessDecodeSupervisedGatedGCNRequest(
        "GatedGCN e-p-d on diff. train sizes, learn rates, train lengths", size, lr, epochs,
        iterations_in_epoch=100, hidden_dim=32)
        for size in [20, 25, 35, 50] for lr in [1e-4] for epochs in [100, 500, 1000]]

    gated_GCN_reinforced_various_requests = [GatedGCNEmbedProcessERReinforcedRequest(
        "GatedGCN emb-proc on diff. train sizes, learn rates, train lengths", size, 0.8, lr, epochs,
        iterations_in_epoch=100, hidden_dim=32, batch_size=1)
        for size in [20, 25, 35, 50] for lr in [1e-4] for epochs in [100, 500, 1000]]

    failed_to_complete = [GatedGCNEmbedProcessERReinforcedRequest(
        "GatedGCN emb-proc that broke down", size, 0.8, lr, epochs,
        iterations_in_epoch=100, hidden_dim=32, batch_size=1)
        for size in [50] for lr in [1e-4] for epochs in [1000]]

    attention_with_edge_features_request = [EncodeProcessDecodeSupervisedAttentionWithEdgeFeaturesRequest(
        "Attention e-p-d test", size, 1e-4, train_epochs, 100, 32) for size in [10, 20] for train_epochs in [500, 1000]]

    batch_testing_currently_best_models_request = [BatchEncodeProcessDecodeSupervisedTrainRequest(
        f"{batch_size}-size minibatches in top e-p-d models, {__train_epochs} epochs", size, 1e-4, train_epochs=__train_epochs,
        iterations_in_epoch=100, hidden_dim=32, batch_size=batch_size)
        for size in [25, 30] for batch_size in [2, 4, 8] for __train_epochs in [500, 2000]]

    deep_message_nets_in_supervised_models = [BatchEncodeProcessDecodeSupervisedDeepMessagesRequest(
        f"3-layer deep NN when constructing messages", 25, 1e-4, epochs, 100, 32, batch_size, processor_depth=depth)
        for epochs in [100, 500] for batch_size in [2, 4, 8] for depth in [2, 5, 7]]

    large_batches_currently_best_models = [BatchEncodeProcessDecodeSupervisedTrainRequest(
        f"{batch_size}-size minibatches in top e-p-d models, {__train_epochs} epochs", size, learn_rate=learn_rate,
        train_epochs=__train_epochs, iterations_in_epoch=100, hidden_dim=32, batch_size=batch_size)
        for size in [25, 30] for batch_size in [16, 32] for __train_epochs in [500, 2000] for learn_rate in [1e-3, 1e-4]]

    deep_message_nets_in_supervised_models_longer_train_and_proc_depth = \
        [BatchEncodeProcessDecodeSupervisedDeepMessagesRequest(
            f"3-layer deep messages, {depth}-l deep processor, {batch_size}-batch", 25, 1e-4, epochs, 100, 32,
            batch_size, processor_depth=depth)
            for epochs in [2000] for batch_size in [4, 8] for depth in [7, 9]]

    batch_reinforced_requests = [
        GatedGCNEmbedProcessERReinforcedRequest("Batch simulation (fixed) in REINFOCE train", size, 0.8, 1e-4, epochs, 20, 32,
                                     embedding_depth=3, processor_depth=3, batch_size=batch_size)
        for size in [20, 25] for epochs in [20, 100, 500] for batch_size in [4, 8, 16]]

    batch_reinforce_embedding_depth_request = [
        GatedGCNEmbedProcessERReinforcedRequest(f"Batch REINFORCE train, {depth}-deep embed, {processor_depth}-deep processor",
                                     20, 0.8, 1e-4, epochs, 100, 32,
                                     embedding_depth=depth, processor_depth=processor_depth, batch_size=8)
        for epochs in [100, 500] for depth in [5, 7, 10] for processor_depth in [1, 3]]

    promising_batch_reinforced_learning_request = [
        GatedGCNEmbedProcessERReinforcedRequest(f"Promising batch REINFORCE model training progress",
                                     20, 0.8, 1e-4, epochs, 100, 32,
                                     embedding_depth=3, processor_depth=3, batch_size=8)
        for epochs in [25, 50, 100, 200, 500, 1000]]

    batched_reinforce_with_deep_embed_and_proc_request = [
        GatedGCNEmbedProcessERReinforcedRequest(f"Promising batch REINFORCE model training progress",
                                     20, 0.8, 1e-4, epochs, 100, 32,
                                     embedding_depth=5, processor_depth=5, batch_size=16)
        for epochs in [25, 50, 100, 200, 500]]

    averaged_simulations_reinforced_leraning_request = [
        AveragingGatedEPReinforceRequest(f"Trying out averaging out simulations for {epochs} epochs",
                                         20, 0.8, 1e-4, epochs, 100, 32, 8) for epochs in [200, 500]
    ]

    __train_graph_size = 10
    __train_epochs = 500
    __iterations_in_epoch = 100
    __TEST_REQUESTS_TO_SUBMIT = [
        BatchEncodeProcessDecodeSupervisedTrainRequest("TESTING basic supervised", __train_graph_size, 1e-4, __train_epochs,
                                                  __iterations_in_epoch, 32),
        EncodeProcessDecodeSupervisedMaskNeighborsRequest("TESTING neighbor masked supervised", __train_graph_size,
                                                          1e-4, __train_epochs, __iterations_in_epoch, 32),
        EncodeProcessDecodeSupervisedGatedGCNRequest("TESTING gated supervised", __train_graph_size, 1e-4,
                                                     __train_epochs, __iterations_in_epoch, 32),
        EncodeProcessDecodeSupervisedAttentionWithEdgeFeaturesRequest(
            "TESTING supervised with attention and edge features", __train_graph_size, 1e-4, __train_epochs, __iterations_in_epoch, 32),
        BatchEncodeProcessDecodeSupervisedTrainRequest("TESTING batch in supervised", __train_graph_size, 1e-4, __train_epochs,
                                                  __iterations_in_epoch, 32, 8),
        BatchEncodeProcessDecodeSupervisedDeepMessagesRequest("TESTING deep messages supervised", __train_graph_size,
                                                              1e-4, __train_epochs, __iterations_in_epoch, 32, 8),
        EmbeddingProcessSupervisedTrainRequest("TESTING embed process supervised", __train_graph_size, 1e-4, __train_epochs,
                                               __iterations_in_epoch, 32),
        BatchEmbeddingProcessErdosRenyiReinforcedLogRequest("TESTING basic reinforcement", __train_graph_size, 0.8, 1e-4,
                                                       __train_epochs, __iterations_in_epoch, 32),
        BatchEmbeddingProcessErdosRenyiReinforcedLogRequest("TESTING batch reinforcement", __train_graph_size, 0.8,
                                                            1e-4, __train_epochs, __iterations_in_epoch, 32, 8),
        GatedGCNEmbedProcessERReinforcedRequest("TESTING gated reinforced", __train_graph_size, 0.8, 1e-4, __train_epochs,
                                                __iterations_in_epoch, 32, 8)
    ]

    basic_supervised_benchmarks = [
        BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"Benchmark (fixed) supervised, {batch_size}-batch epd, {train_size}-train size", train_size, 1e-4, epochs, 100, 32, batch_size)
        for epochs in [25, 50, 100, 500, 2000] for batch_size in [1, 8, 32] for train_size in [15, 25, 30]]

    other_supervised_benchmarks = [
        [EncodeProcessDecodeSupervisedGatedGCNRequest(
            "Benchmark (fixed) gated supervised epd", 25, 1e-4, epochs, 100, 32, 8)
            for epochs in [1000]],
        [BatchEncodeProcessDecodeSupervisedDeepMessagesRequest(
        "Benchmark (fixed) deep messages supervised epd", 25, 1e-4, epochs, 100, 32, 8)
            for epochs in [100, 1000]],
        [EmbeddingProcessSupervisedTrainRequest(
            "Benchmark (fixed) embed-process supervised", 25, 1e-4, epochs, 100, 32, batch_size=32,
            embedding_depth=7, processor_depth=2) for epochs in [100, 1000]],
        ]

    gated_reinforcement_benchmark = [
        GatedGCNEmbedProcessERReinforcedRequest(
            f"Benchmark (fixed) gated reinforced {batch_size}-batch", train_size, 0.8, 1e-4, epochs, 100, 32, batch_size)
        for epochs in [25, 50, 100, 1000] for batch_size in [1, 8, 32] for train_size in [20, 25, 30]]

    benchmark_requests = basic_supervised_benchmarks \
                         + [r for request_class in other_supervised_benchmarks for r in request_class] \
                         + gated_reinforcement_benchmark

    gated_reinforcement_models_no_retries = [
        GatedGCNEmbedProcessERReinforcedRequest(
            f"Gated reinforcement with only a SINGLE try per graph. {train_size}-train_size",
            train_size, 0.8, 1e-4, epoch, 100, 32)
        for train_size in [20, 25, 30] for epoch in [100, 500, 2000]]

    different_supervised_train_graphs_requests = [
        BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"MAIN supervised model, {noise} train graph noise", 25, 1e-4, 2000, 100, 32, 8,
            nr_expected_noise_edges_per_node=noise) for noise in [0.1, 0.5, 1, 1.5, 3, 5, 10]
    ]

    main_models = [
        BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"MAIN supervised model, {train_size} train size", train_size, 1e-4, 2000, 100, 32, 8,
            nr_expected_noise_edges_per_node=3, save_train_history=True)
        for train_size in [25, 27, 30]
    ]

    minimal_benchmark_for_testing_purposes = [
        GatedGCNEmbedProcessERReinforcedRequest(
            f"Minimal benchmark gated reinforce test", 5, 0.8, 1e-4, 5, 10, 32, 2),
        BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"Minimal benckmark supervised test", 5, 1e-4, 5, 10, 32, 2)]

    decay_rate_supervised_models = [
        DecayingLRBatchEPDSupervisedLogRequest(
            f"Decaying learn rate for supervised models, {depth}-processor depth", train_size, 1e-3, 2000, 100,
            32, 8, nr_expected_noise_edges_per_node=3, processor_depth=depth)
        for train_size in [25] for depth in [5, 8, 15]
    ]

    processor_depth_supervised_models = [
        BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"MAIN supervised model, {train_size} train size, {processor_depth}", train_size, 1e-4, 2000, 100, 32, 8, nr_expected_noise_edges_per_node=3, processor_depth=processor_depth)
        for train_size in [25] for processor_depth in [1, 3, 5, 8, 12, 15]
    ]

    another_round_of_gated_ep_supervised = [EncodeProcessDecodeSupervisedGatedGCNRequest(
        f"GatedGCN e-p-d supervised. Size {size}, {epochs} epochs", size, lr, epochs, 100, 32, 8, nr_expected_noise_edges_per_node=3)
        for size in [20, 25] for lr in [1e-4] for epochs in [100, 500, 1000]]

    gated_reinforcement_depth_test = [
        GatedGCNEmbedProcessERReinforcedRequest(
            f"lr={lr}, Edge+node residual, LkyReLU(0.2 neg sl.) only!, softmax, emb_depth={depth}, gated reinforced", train_size,
            0.8, lr, epochs, 100, 32, embedding_depth=depth, save_train_history=True)
        for epochs in [1000] for train_size in [25] for depth in [3, 5, 8] for lr in [3*1e-5, 1e-5]]

    main_supervised_models_decay_tryout = [
        DecayingLRBatchEPDSupervisedLogRequest(
            f"MAIN supervised model with decay, {train_size} train size", train_size, 1e-4, 2000, 100, 32, 8,
            nr_expected_noise_edges_per_node=3, save_train_history=True)
        for train_size in [25, 27, 30]
    ]

    gated_reinforcement_regularization_test = [
        GatedGCNEmbedProcessERReinforcedRequest(
            f"Sigmoid Gated reinforced, regularization weight={reg_w}", train_size, 0.8, 1e-4, epochs, 100, 32,
            embedding_depth=3, loss_l2_regularization_weight=0, save_train_history=True)
        for epochs in [1000] for train_size in [25] for reg_w in [0, 0.001, 0.01, 0.1, 1]]

    # ----DID NOT TRY OUT REQUESTS BELOW---------------------------------------------------------------

    PAPER_MODELS = [
        BatchEncodeProcessDecodeSupervisedTrainRequest(
            f"Final paper supervised model", 25, 1e-4, 2000, 100, 32, 8,
            nr_expected_noise_edges_per_node=3, save_train_history=True),
        GatedGCNEmbedProcessERReinforcedRequest(
            f"Paper model! lr=1e-5, softmax, emb_depth=8, gated reinforced",
            25,
            0.8, 1e-5, 1000, 100, 32, embedding_depth=8, save_train_history=True)
    ]

    REQUESTS_TO_SUBMIT = PAPER_MODELS

    # torch.set_num_threads(1)
    # for r in __TEST_REQUESTS_TO_SUBMIT[7:8]:
    #     print(f"Starting {r.short_description}")
    #     r.run_log_request(database_lock)

    with mp.Pool(min(NR_THREADS, len(REQUESTS_TO_SUBMIT)), initializer=pool_initializer, initargs=(database_lock,)) as p:
        results = p.map_async(log_request_starter, REQUESTS_TO_SUBMIT)
        results.get()
