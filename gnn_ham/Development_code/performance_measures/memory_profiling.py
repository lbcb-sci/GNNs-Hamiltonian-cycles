import torch
import torch_geometric as torch_g
import time

from src.Models import EncodeProcessDecodeAlgorithm
from src.GraphGenerators import ErdosRenyiGenerator

if __name__ == '__main__':
    sizes = [500, 1000, 2000]
    start_time = time.time()
    HamS = EncodeProcessDecodeAlgorithm(True, processor_depth=5, hidden_dim=32)
    for num_nodes in sizes:
        print(f"Profiling on size {num_nodes} ({time.time() - start_time})")
        generator = ErdosRenyiGenerator(num_nodes, 0.8)
        d = next(iter(generator))
        HamS.init_graph(d)
        HamS.prepare_for_first_step(d, 0)
        logits = HamS.next_step_logits_masked_over_neighbors(torch_g.data.Batch.from_data_list([d]))
        print(f"Logits_computed ({time.time() - start_time})")

        with torch.no_grad():
            d = next(iter(generator))
            HamS.batch_run_greedy_neighbor(torch_g.data.Batch.from_data_list([d]))
        print(f"Computed greedy route ({time.time() - start_time})")