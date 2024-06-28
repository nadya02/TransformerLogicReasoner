import json
import numpy as np
import os, random
import torch

path = 'data/adalab-meta.adalab-meta-ontology.6.owl_proofs/Input0.json'
dataset_path = 'data'
PAD = 256
EOS = 24
max_length = 512

def create_fixed_batches(data, batch_size, max_seq_length, pad_token):
    inputs, proofs = zip(*data)

    batches = []
    for i in range(0, len(inputs), batch_size):
        batch_input = inputs[i:i+batch_size]
        batch_proof = proofs[i:i+batch_size]

        padded_inputs = np.array([np.pad(input[:max_seq_length],
                                         (0, max(0, max_seq_length - len(input))),
                                         mode='constant', constant_values=0) for input in batch_input])
        padded_proofs = np.array([np.pad(proof[:max_seq_length],
                                         (0, max(0, max_seq_length - len(proof))),
                                         mode='constant', constant_values=0) for proof in batch_proof])
        batches.append((padded_inputs, padded_proofs))
    return batches

def create_tensors(data, batch_size=32, max_seq_length=512, pad_index=0):
    batches_labeled = create_fixed_batches(data, batch_size, max_seq_length, pad_index)
    tensor_batches = []
    padding_mask = []

    for padded_inputs, batch_proofs in batches_labeled:
        input_tensor = torch.tensor(padded_inputs, dtype=torch.long)
        proof_tensor = torch.tensor(batch_proofs, dtype=torch.long)

        # Create the padding masks for inputs and outputs
        input_padding_mask = (input_tensor != pad_index).long()
        output_padding_mask = (proof_tensor != pad_index).long()

        tensor_batches.append((input_tensor, proof_tensor))
        padding_mask.append((input_padding_mask, output_padding_mask))

    tensor_batches = list(tensor_batches)

    input, target = zip(*tensor_batches)
    last_el_idx = len(input) - 1
    if(len(tensor_batches[last_el_idx][0]) < batch_size):
        print(f"Last batch should be droped!, {tensor_batches[last_el_idx][0]}")
        tensor_batches.pop()

    print(f"Last idx: {last_el_idx}, batch size: {batch_size}")
    # print(f"last element: {input[last_el_idx]}")
    return tensor_batches, padding_mask

# def json_to_string(input_files, proof_files, ontology_path):

#     premises, proofs = [], []
#     for input_f, proof_f, in zip(input_files, proof_files):
#       with open(os.path.join(ontology_path, input_f), 'r') as f:
#           premise_string = json.dumps(json.load(f)).encode('utf-8')
#           encoded_premise = np.frombuffer(premise_string, dtype=np.uint8)

#       with open(os.path.join(ontology_path, proof_f), 'r') as f:
#           proof_string = json.dumps(json.load(f)).encode('utf-8')
#           encoded_proof = np.frombuffer(proof_string, dtype=np.uint8)

#       premises.append(encoded_premise)
#       proofs.append(encoded_proof)

#     return premises, proofs

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


def load_ontology(path, train_ratio = 0.7, val_ratio = 0.1):

    train_data = []
    val_data = []
    test_data = []

    print(f"Path: {path}")
    premises, proofs = [], []
    for dir in os.listdir(path):
        num_inputs = 0
        num_proofs = 0

        ontology_path = os.path.join(path, dir)

        input_files = sorted([f for f in os.listdir(ontology_path) if f.startswith("Input")])
        proof_files = sorted([f for f in os.listdir(ontology_path) if f.startswith('Proof')])
        cur_dir_premises, cur_dir_proofs =  json_to_string(input_files, proof_files, ontology_path)

        premises.extend(cur_dir_premises)
        proofs.extend(cur_dir_proofs)

        print(f"Current dir: {dir}: {len(cur_dir_premises)}")
        assert len(premises) == len(proofs)

    data_length = len(proofs)
    print(f"Proof length before split: {data_length}")
    train_len = int(data_length * train_ratio)
    val_len = int(train_len * val_ratio)

    # Gives the indices from 0 to data_length and then shuffles them
    indices = np.arange(data_length)
    # np.random.shuffle(indices)

    # Using indices to split the data
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]

    train_data = [(premises[i], proofs[i]) for i in train_indices]
    val_data = [(premises[i], proofs[i]) for i in val_indices]
    test_data = [(premises[i], proofs[i]) for i in test_indices]

    print(type(train_data))
    return train_data, val_data, test_data