config = {
    'pad_index': 0,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'num_epochs': 1000,
    'optimizer': 'Adam',
    'lr_warmup': 10000,
    'device': 'cuda',  # 'cuda' or 'cpu'
    'seed': 42,
    'embed_dim': 256,
    'heads': 4,
    'd_ff': 256,
    'seq_len': 512,
    'N': 8,
    'num_tokens': 256,
    'context': 512,
    'char_to_gen': 512,
    'log_interval': 100,
    'save_interval': 1000,
    'compression_interval': 1000,
    'log_dir': 'logs/',
    'model_dir': 'saved_models/',
    'data_path': 'data_direct_true_100_ontologies',
}

def get_device():
    import torch
    return torch.device(config['device'] if torch.cuda.is_available() else 'cpu')