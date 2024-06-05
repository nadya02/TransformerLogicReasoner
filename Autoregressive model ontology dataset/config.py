config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_batches': 40000,
    'optimizer': 'Adam',
    'lr_warmup': 10000,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'seed': 42,
    'embed_dim': 128,
    'heads': 4,
    'd_ff': 256,
    'seq_len': 256,
    'N': 8,
    'num_tokens': 256,
    'context': 256,
    'char_to_gen': 128,
    'test_interval': 100,
    'log_dir': 'logs/',
    'model_dir': 'saved_models/',
    'data_path': 'data',
}

def get_device():
    import torch
    return torch.device(config['device'] if torch.cuda.is_available() else 'cpu')