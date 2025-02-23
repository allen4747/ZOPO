import random
SEED=0

def subsample_data_(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    # random.seed(SEED)
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    while True:
        indices = random.sample(range(len(inputs)), subsample_size)
        _inputs = [inputs[i] for i in indices]
        _outputs = [outputs[i] for i in indices]
        if len(set(_outputs)) > 1:
            break
    return _inputs, _outputs

def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    # random.seed(SEED)
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    _inputs = [inputs[i] for i in indices]
    _outputs = [outputs[i] for i in indices]
    return _inputs, _outputs


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    random.seed(0)
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)
