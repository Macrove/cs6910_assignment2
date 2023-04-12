import wandb
from main import main
from params.default_params import optimizer_param_map
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
    
    main(epochs, activation, cnn_params, out_features_fc1, dropout, loss, lr, optimizer, batch_normalisation=True, use_wandb=True)

sweep_configuration = {
    "name": "Adam Sweep",
    "method": "bayes",
    "metric": {'goal': 'maximize', 'name': 'val_acc'},
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3
     },
    "parameters": {
        'epochs': {'values': [8, 9, 10, 11, 12]},
        'activation': {'values': ["ReLU", "GELU", "SiLU", "Mish"]},
        'out_features_fc1': {'values': [256, 512, 1024, 2048]},
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4]},
        'lr': {'min': 0.0001, 'max': 0.5},
        'optimizer': {'values' :['Adam']},
        'beta1': {'min': 0.6, 'max': 0.99},
        'beta2': {'min': 0.7, 'max': 0.9999},
        'momentum': {'values': [1e-1, 1e-2, 1e-3]},
        'loss': {'values': ['cross_entropy']},
        'n_filters': {'values': [32, 64]},
        'filter_organisation': {'values': [0, 1]},
        'kernel_size': {'values': [2]}
    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-2", entity="me19b110")
wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=15)
