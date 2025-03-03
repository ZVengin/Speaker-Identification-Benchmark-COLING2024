# Generate BERT features.

class InputFeatures(object):
    """
    Inputs of the BERT model.
    """
    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(batch, tokenizer):
    """
    Convert textual segments into word IDs.

    params
        examples: the raw textual segments in a list.
        tokenizer: a BERT Tokenizer object.

    return
        features: BERT features in a list.
    """
    features_batch = []
    max_seq_len = 0
    for examples in batch:
        features = []
        for (ex_index, example) in enumerate(examples):
            # tokens = tokenizer.tokenize(example)
            tokens = example

            new_tokens = []
            input_type_ids = []

            new_tokens.append("[CLS]")
            input_type_ids.append(0)
            new_tokens += tokens
            input_type_ids += [0] * len(tokens)
            new_tokens.append("[SEP]")
            input_type_ids.append(0)

            # print(f'token length:{len(tokens)}')
            # print(f'tokens:{tokens}')

            input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
            input_mask = [1] * len(input_ids)

            features.append(
                InputFeatures(
                    tokens=new_tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
        features_batch.append(features)
        seq_lens = list(map(lambda x:len(x.tokens),features))
        if max(seq_lens)>max_seq_len:
            max_seq_len = max(seq_lens)
    for features in features_batch:
        for feature in features:
            feature.input_ids = feature.input_ids + [tokenizer.mask_token_id] * (max_seq_len - len(feature.tokens))
            feature.input_mask = feature.input_mask + [0] * (max_seq_len - len(feature.tokens))

    return features_batch
