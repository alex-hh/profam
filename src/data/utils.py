def examples_to_list_of_dicts(examples):
    keys = list(examples.keys())
    return [{k: examples[k][i] for k in keys} for i in range(len(examples[keys[0]]))]


def examples_list_to_dict(examples):
    keys = list(examples[0].keys())
    return {k: [example[k] for example in examples] for k in keys}