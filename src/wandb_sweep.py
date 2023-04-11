import wandb
from main import main
from params.default_params import optimizer_param_map
from utils.maps import activation_map
from utils.prepare_params import get_cnn_params

def run_sweeps():
    run = wandb.init()
    config = wandb.config
    run.name = "epochs_{}".format(config.epochs)

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
    "name": "first_sweep",
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
        'lr': {'values': [1e-1, 1e-2, 1e-3]},
        'optimizer': {'values' :['SGD', 'Adam']},
        'beta1': {'values': [0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]},
        'beta2': {'values': [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]},
        'momentum': {'values': [1e-1, 1e-2, 1e-3]},
        'loss': {'values': ['cross_entropy']},
        'n_filters': {'values': [32, 64, 128]},
        'filter_organisation': {'values': [0, 1, 2]},
        'kernel_size': {'values': [2, 3, 4, 5]}
    }
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-2", entity="me19b110")
wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=1)

##########################################################################################################
# code for optimizer wise sweeps

# sweep_configuration_map = {
#     "nadam": {
#         "name": "nadam_sweep",
#         "method": "random",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [10, 11, 12, 13, 14]},
#             'n_hidden_layers': {'values': [2, 3, 4]},
#             'size_hidden_layer': {'values': [128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 8 * 1e-5, 'max': 1 * 1e-3},
#             'optimizer': {'values' :['adam']},
#             'batch_size': {'values': [16, 32, 64, 128]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.8},
#             'beta1': {'min': 0.7, 'max': 0.8},
#             'beta2': {'min': 0.8, 'max': 0.95},
#             'epsilon': {'values': [1e-10]},
#             'gamma': {'values': [1e-5]},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "adam": {
#         "name": "adam_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [8, 9, 10, 11, 12, 13, 14]},
#             'n_hidden_layers': {'values': [2, 3, 4, 5]},
#             'size_hidden_layer': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-4},
#             'optimizer': {'values' :['adam']},
#             'batch_size': {'values': [16, 32, 64, 128]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.8},
#             'beta1': {'min': 0.6, 'max': 0.8},
#             'beta2': {'min': 0.7, 'max': 0.9},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "rmsprop": {
#         "name": "rmsprop_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [10, 11, 12, 13]},
#             'n_hidden_layers': {'values': [2, 3, 4, 5]},
#             'size_hidden_layer': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [0, 1e-6, 1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-4},
#             'optimizer': {'values' :['rmsprop']},
#             'batch_size': {'values': [16, 32, 64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.4, 'max': 0.9},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "nag": {
#         "name": "nag_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [10, 11, 12]},
#             'n_hidden_layers': {'values': [2, 3, 4, 5]},
#             'size_hidden_layer': {'values': [64, 128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-5},
#             'optimizer': {'values' :['nag']},
#             'batch_size': {'values': [64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.99},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "momentum": {
#         "name": "momentum_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [8, 9, 10, 11]},
#             'n_hidden_layers': {'values': [2, 3, 4]},
#             'size_hidden_layer': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [0, 1e-6, 1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-4},
#             'optimizer': {'values' :['momentum']},
#             'batch_size': {'values': [16, 32, 64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.99},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'values': [1e-1, 1e-2, 1e-3]},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "sgd": {
#         "name": "sgd_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [8, 9, 10, 11, 12]},
#             'n_hidden_layers': {'values': [4, 5]},
#             'size_hidden_layer': {'values': [64, 128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-5},
#             'optimizer': {'values' :['sgd']},
#             'batch_size': {'values': [64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['ReLU']},
#             'beta': {'min': 0.5, 'max': 0.99},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration_map["nadam"], project="cs6910-assignment-1", entity="me19b110")
# wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=5)
# for key in sweep_configuration_map.keys():
#     sweep_id = wandb.sweep(sweep=sweep_configuration_map[key], project="cs6910-assignment-1", entity="me19b110")
#     wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=20)

