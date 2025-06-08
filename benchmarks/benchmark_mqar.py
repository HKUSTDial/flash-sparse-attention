from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig


seq_len = 2048
d_model = 256  # 32, 64, 128, 256
vocab_size = seq_len + 1
num_kv_pairs = 512

if d_model == 32:
    learning_rate = 4e-4
elif d_model == 64:
    learning_rate = 3e-4
elif d_model == 128:
    learning_rate = 2e-4
elif d_model == 256:
    learning_rate = 1e-4

if seq_len == 1024:
    batch_size = 64
elif seq_len == 2048:
    batch_size = 32
elif seq_len == 4096:
    batch_size = 16
elif seq_len == 8192:
    batch_size = 8

config = TrainConfig(
    learning_rate=learning_rate,
    data=DataConfig(
        cache_dir=".cache",
        batch_size=batch_size,
        train_configs=[
            MQARConfig(
                num_examples=250_000,
                vocab_size=vocab_size,
                input_seq_len=seq_len,
                num_kv_pairs=num_kv_pairs,
            )
        ],
        test_configs=[
            MQARConfig(
                num_examples=1_000,
                vocab_size=vocab_size,
                input_seq_len=seq_len,
                num_kv_pairs=num_kv_pairs,
            )
        ]
    ),
    model=ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        max_position_embeddings=seq_len,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.dma.DynamicMaskAttention",
            kwargs={"keep_window_size": num_kv_pairs, "num_heads": 1},
        )
    ),
    
)

configs = [config]