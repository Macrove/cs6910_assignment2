default_model_params = {
    "optimizer" : "SGD",
    "epochs" : 20,
    "linear_fc_out_features" : 512,
    "activation" : "ReLU",
    "loss": "cross_entropy",
    "n_filters": 64,
    "filter_organisation": 1,
    "dropout": 0,
    "kernel_size": 2,
    "stride": 1,
    "padding": 0,
    "batch_normalisation": 1,
    "init": "xavier"
}

default_use_wandb = 1


default_credentials = {
    "wandb_project": "cs6910-assignment-2",
    "wandb_entity": "me19b110"
}

optimizer_param_map = {
    "SGD" : {
        "name": "SGD",
        "default_params": dict(
            lr = 0.01,
            momentum= 0.01,
            # dampening= 0.1,
            # weight_decay= 1e-3,
            # nestrov= True
        )
    },
    "Adam" : {
        "name": "Adam",
        "default_params": dict(
            lr= 0.001,
            betas= (0.9, 0.99),
        )
    }
}