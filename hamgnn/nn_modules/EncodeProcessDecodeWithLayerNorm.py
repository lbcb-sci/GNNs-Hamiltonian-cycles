import hamgnn.nn_modules.EncodeProcessDecodeNN as EncodeProcessDecodeNN
import hamgnn.nn_modules.nn_modules as nn_modules

class EncodeProcessDecodeWithLayerNorm(EncodeProcessDecodeNN.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return nn_modules.ResidualMultilayerMPNNLayerNorm(
            dim=self.hidden_dim, message_dim=self.hidden_dim, edges_dim=1, nr_layers=self.processor_depth)
