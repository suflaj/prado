import torch


def pad_projections(batch):
    tokens, labels = zip(*batch)

    max_token_length = max(len(token_list) for token_list in tokens)
    tokens = [
        token_list + [""] * (max_token_length - len(token_list))
        for token_list in tokens
    ]

    return tokens, torch.tensor(labels)
