default_model_params = {
    "optimizer" : "sgd",
    "epochs" : 40,
    "linear_fc_out_features" : 1024,
    "activation" : "ReLU",
    "loss": "cross_entropy",
    "n_filters": 32,
    "filter_organisation": 1,
    "dropout": 0.1,
    "kernel_size": 2,
    "stride": 1,
    "padding": 1,
    "batch_normalisation": 1
}

default_use_wandb = False

# default_dataset = 'fashion_mnist'

default_credentials = {
    "wandb_project": "cs6910-assignment-2",
    "wandb_entity": "me19b110"
}

optimizer_param_map = {
    "sgd" : {
        "name": "sgd",
        "default_params": {
            "learning_rate" : 0.01
        }
    },
    "momentum" : {
        "name": "momentum",
        "default_params": {
            "eta" : 0.000001,
            "gamma": 0.0000039
        }
    },
    "nag" : {
        "name": "nag",
        "default_params": {
            "eta" : 0.0000086,
            "gamma": 0.0000021
        }
    },
    "rmsprop" : {
        "name": "rmsprop",
        "default_params": {
            "eta" : 0.00001,
            "beta": 0.75,
            "epsilon": 1e-10
        }
    },
    "adam" : {
        "name": "adam",
        "default_params": {
            "eta" : 0.00001,
            "beta1": 0.7483,
            "beta2": 0.7838,
            "epsilon": 1e-9
        }
    },
    "nadam" : {
        "name": "nadam",
        "default_params": {
            "eta" : 0.00008,
            "beta1": 0.7803,
            "beta2": 0.89504,
            "epsilon": 1e-9
        }
    }
}