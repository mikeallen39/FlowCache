from .modeling import MagiAttention_init

def replace_magi(compression_config):
    from inference.model.dit import VideoDiTModel, FullyParallelAttention
    def init_wrapper(self, model_config, engine_config, layer_number):
        MagiAttention_init(self, model_config, engine_config, layer_number, compression_config)

    FullyParallelAttention.__init__ = init_wrapper