import torch
from datetime import datetime

from utils import *
from model import TransformerModel
from data_loader import *   
from config import config, get_device
from torch.optim.lr_scheduler import LambdaLR

import wandb
# api_key = "4556651ce2176ed46d63906314d5c7703d3ff741"
# wandb.login(key=api_key)
# wandb.init(project="transformer_training", config=config)

def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train(config):
    torch.manual_seed(config['seed'])
    device = get_device()

    # train_data, val_data, test_data = load_ontology(config['data_path'])
    # train_batches = create_tensors(train_data[0], train_data[1], config['batch_size'], config['seq_len'], config['pad_index'])
    # val_batches = create_tensors(val_data[0], val_data[1], config['batch_size'], config['seq_len'], config['pad_index'])
    # test_batches = create_tensors(test_data[0], test_data[1], config['batch_size'], config['seq_len'], config['pad_index'])

    model = TransformerModel(embed_dim=config['embed_dim'], heads=config['heads'], d_ff=config['d_ff'],
                        seq_len=config['seq_len'], N=config['N'], num_tokens=config['num_tokens']).to(device)

    optimizer = get_optimizer(config['optimizer'], model.parameters(), config['learning_rate'])

    model.train()
    for epoch in tqdm.trange(config['num_epochs']):
      for batch_idx, ((inputs, targets), (input_padding_mask, output_padding_mask)) in enumerate(zip(train_batches, train_masks)):
          inputs, targets = inputs.to(device), targets.to(device)
          input_padding_mask = input_padding_mask.to(device)
          output_padding_mask = output_padding_mask.to(device)
          optimizer.zero_grad()

          # if torch.cuda.is_available():
          #     input, output = input.cuda(), output.cuda()

          outputs = model(inputs, padding_mask=input_padding_mask)
          loss = F.nll_loss(outputs.transpose(2, 1), targets, ignore_index=config['pad_index'], reduction='mean')
          loss.backward()
          total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

          # wandb.log({"Loss/train": loss.item(),
          #     "Gradient norm": total_norm,
          #     "Epoch": epoch})

          optimizer.step()

          if batch_idx % config['log_interval'] == 0:
              print(f"Batch {batch_idx}, Loss: {loss.item()}")
              random_idx = random.randint(0, len(val_data))
              val_inputs, val_targets = zip(*val_data)
              random_val = val_inputs[random_idx]

              val_input = val_inputs[random_idx]
              seed = torch.tensor(random_val, dtype=torch.long).to(device)
              # seed = torch.tensor(selected_input, dtype=torch.long)

              print_sequence(model, seed, config['char_to_gen'], config['context'])

              # Estimate compression on a subset of the validation data
              # val_data_2 = torch.cat([val_input for val_input, _ in val_batches])
              # val_data_2 = torch.cat([batch_inputs for batch_inputs, _ in val_batches], dim=0)
              # nsamples = min(100, val_data_2.size(0))
              # print("Nsamples: ", nsamples)
              # bits_per_byte = estimate_compression(model, val_data_2, nsamples, context=config['context'], batch_size=config['batch_size'])
              # print(f"Bits per byte: {bits_per_byte}")

          if batch_idx % config['save_interval'] == 0:
                  torch.save(model.state_dict(), f'model_epoch_{epoch}_batch_{batch_idx}.pt')
            
train_data, val_data, test_data = load_ontology(config['data_path'])
train_batches, train_masks = create_tensors(train_data, config['batch_size'], config['seq_len'], config['pad_index'])
val_batches, val_masks = create_tensors(val_data, config['batch_size'], config['seq_len'], config['pad_index'])
test_batches, test_masks = create_tensors(test_data, config['batch_size'], config['seq_len'], config['pad_index'])

train(config)