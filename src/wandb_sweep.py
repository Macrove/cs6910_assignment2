import wandb
from main import main
from params.default_params import optimizer_param_map, default_model_params
from utils.maps import activation_map
from utils.prepare_params import get_cnn_params

def run_sweeps():
    run = wandb.init()
    config = wandb.config
    run.name = "opt_{}_lr_{:.3f}".format(config.optimizer, config.lr)

    config.betas = (config.beta1, config.beta2)
    optimizer = optimizer_param_map[config.optimizer]
    for key in optimizer["default_params"].keys():
        optimizer["default_params"][key] = getattr(config, str(key))

    epochs = config.epochs
    activation = config.activation
    out_features_fc1 = config.out_features_fc1
    dropout = config.dropout
    lr = config.lr
    loss = config.loss
    n_filters = config.n_filters
    filter_organisation = config.filter_organisation
    kernel_size = config.kernel_size

    cnn_params = get_cnn_params(n_filters, filter_organisation, kernel_size)
    
    init = default_model_params["init"]

    main(epochs, activation, cnn_params, out_features_fc1, dropout, loss, lr, optimizer, batch_normalisation=True, init=init, use_wandb=True)

sweep_configuration = {
    "name": "SGD Sweep2",
    "method": "bayes",
    "metric": {'goal': 'maximize', 'name': 'val_acc'},
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3
     },
    "parameters": {
        'epochs': {'values': [10]},
        'activation': {'values': ["ReLU"]},
        'out_features_fc1': {'values': [512]},
        'dropout': {'values': [0]},
        'lr': {'min': 0.001, 'max': 0.1},
        'optimizer': {'values' :['SGD']},
        'beta1': {'min': 0.6, 'max': 0.61},
        'beta2': {'min': 0.6, 'max': 0.61},
        'momentum': {'min': 0.5, 'max': 0.9},
        'loss': {'values': ['cross_entropy']},
        'n_filters': {'values': [64]},
        'filter_organisation': {'values': [1]},
        'kernel_size': {'values': [2]}
    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-2", entity="me19b110")
wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=10)
