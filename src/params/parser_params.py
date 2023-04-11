from utils.maps import activation_map
parser_choices = {
    "wandb_project": ["cs6910-assignment-2"],
    "wandb_entity": ["me19b110"],
    "loss": ["cross_entropy"],
    "optimizer": ["sgd"],
    "activation": [fn for fn in activation_map.keys()],
    "filter_organisation": [0, 1, 2]
}