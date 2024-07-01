import json
import numpy as np
import os, random
import torch
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

path = 'data/adalab-meta.adalab-meta-ontology.6.owl_proofs/Input0.json'
dataset_path = 'data'
PAD = 256
EOS = 24
max_length = 512

def collate_fn(batch):
    premises = [torch.tensor(item['premise'], dtype=torch.long) for item in batch]
    proofs = [torch.tensor(item['proof'], dtype=torch.long) for item in batch]
    
    premises_padded = pad_sequence(premises, batch_first=True, padding_value=0)
    proofs_padded = pad_sequence(proofs, batch_first=True, padding_value=0)
    
    # Create attention masks
    premises_masks = (premises_padded != 0).long()
    proofs_masks = (proofs_padded != 0).long()
    
    return premises_padded, proofs_padded, premises_masks, proofs_masks

def json_to_string(input_files, proof_files, ontology_path):
    premises, proofs = [], []
    for input_f, proof_f in zip(input_files, proof_files):
        # Attempt to process the input file
        input_file_path = os.path.join(ontology_path, input_f)
        try:
            with open(input_file_path, 'r') as f:
                premise_string = json.dumps(json.load(f)).encode('utf-8')
                encoded_premise = np.frombuffer(premise_string, dtype=np.uint8)
            premises.append(encoded_premise)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {input_file_path}: {str(e)}")
            # Optionally, raise the error to stop the process or handle it appropriately
            # raise

        # Attempt to process the proof file
        proof_file_path = os.path.join(ontology_path, proof_f)
        try:
            with open(proof_file_path, 'r') as f:
                proof_string = json.dumps(json.load(f)).encode('utf-8')
                encoded_proof = np.frombuffer(proof_string, dtype=np.uint8)
            proofs.append(encoded_proof)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {proof_file_path}: {str(e)}")
            # Optionally, raise the error to stop the process or handle it appropriately
            # raise

    return premises, proofs


def load_ontology(path, train_ratio=0.7, val_ratio=0.1):
    premises, proofs = [], []
    
    print(f"Path: {path}")
    for dir in os.listdir(path):
        ontology_path = os.path.join(path, dir)
        input_files = sorted([f for f in os.listdir(ontology_path) if f.startswith("Input")])
        proof_files = sorted([f for f in os.listdir(ontology_path) if f.startswith('Proof')])
        cur_dir_premises, cur_dir_proofs = json_to_string(input_files, proof_files, ontology_path)
        
        premises.extend(cur_dir_premises)
        proofs.extend(cur_dir_proofs)
        
        print(f"Current dir: {dir}: {len(cur_dir_premises)}")
        assert len(premises) == len(proofs)

    data_length = len(proofs)
    train_len = int(data_length * train_ratio)
    val_len = int(train_len * val_ratio)

    indices = np.arange(data_length)
    np.random.shuffle(indices)

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    train_data = [{"premise": premises[i], "proof": proofs[i]} for i in train_indices]
    val_data = [{"premise": premises[i], "proof": proofs[i]} for i in val_indices]
    test_data = [{"premise": premises[i], "proof": proofs[i]} for i in test_indices]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
