import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import tqdm
import random
import time
import math

LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tics = []

def print_average_length_of_ascii_lists(ascii_lists):
    if not ascii_lists:  # Check if the list is empty
        print("The list is empty. No average length can be calculated.")
        return

    total_length = sum(len(ascii_list) for ascii_list in ascii_lists)  # Sum the lengths of all ASCII lists
    average_length = total_length / len(ascii_lists)  # Calculate the average

    print(f"The average length of the ASCII lists is: {average_length:.2f}")

def slice_batch(data, seq_length, batch_size):
    start_indices = torch.randint(size=(batch_size,), low=0, high=data.size(0) - seq_length - 1)

    sequences_inputs = [data[start_i: start_i + seq_length] for start_i in start_indices]
    sequences_targets = [data[start_i + 1: start_i + seq_length + 1] for start_i in start_indices]

    inputs = torch.cat([seq[None, :] for seq in sequences_inputs], dim=0).to(torch.long)
    targets = torch.cat([seq[None, :] for seq in sequences_targets], dim=0).to(torch.long)

    return inputs, targets

def sample(lnprobs, temperature=1.0):
    print("I'm here")
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def sample_sequence(model, seed, char_to_gen, max_len, temperature=1.0):
    """Sample a sequence from the model based on the seed sequence."""
    model.eval()
    with torch.no_grad():
        sequence = seed.detach().clone()
        for _ in range(char_to_gen):
            input_seq = sequence[-max_len:].unsqueeze(0).to(device)
            output = model(input_seq)
            probabilities = torch.softmax(output[0, -1, :] / temperature, dim=0)
            next_char = torch.multinomial(probabilities, num_samples=1)
            sequence = torch.cat((sequence, next_char), dim=0)
    model.train()
    return sequence

def generate_text(model, seed, char_to_gen, max_len, temperature=1.0):
    """Sample a sequence from the model based on the seed sequence."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        sequence = seed.clone().detach().to(device)
        for _ in range(char_to_gen):
            input_seq = sequence[-max_len:].unsqueeze(0)  # Add batch dimension
            output = model(input_seq)
            probabilities = torch.softmax(output[0, -1, :] / temperature, dim=0)
            next_char = torch.multinomial(probabilities, num_samples=1)
            sequence = torch.cat((sequence, next_char), dim=0)
    model.train()
    return sequence

def print_sequence(model, seed, char_to_gen, context):
    """Print the seed and the generated sequence"""
    seed_list = seed.tolist()

    generated_seq = generate_text(model, seed, char_to_gen, context)

    # Convert the seed and the generated sequence to text
    seed_text = ''.join(chr(idx) for idx in seed_list)
    new_text = generated_seq[len(seed):]
    generated_text = ''.join(chr(idx) for idx in new_text.tolist())

    print(f"SEED: {seed_text}")
    print(f"Generated text: {generated_text}")

    return generated_text
    
def estimate_compression(model, data, nsamples, context, batch_size, verbose=False, model_produces_logits=False):
    bits, tot = 0.0, 0
    batch = []

    # indices of target characters in the data
    gtargets = random.sample(range(data.size(0)), k=nsamples)

    target_indices = []

    for i, current in enumerate(tqdm.tqdm(gtargets) if verbose else gtargets):
        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the context is the input

        target_indices.append(instance.size(0) - 2) # index of the last element of the context

        if instance.size(0) < context + 1:
            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or i == len(gtargets) - 1:
            # batch is full, or we are at the last instance, run it through the model

            b = len(batch)

            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[:, -1]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

                if model_produces_logits:
                    output = F.log_softmax(output, dim=-1)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=device), target_indices, target]
            log2probs = lnprobs * LOG2E
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilties) to the running total
            batch, target_indices = [], []  # clear the buffer

    return bits.item() / nsamples # total nr of bits used

def tic():
    tics.append(time.time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()