# Assignment Repository for CS6910

This repository contains all the code related to assignment 2 of cs6910.<br>
All the code for PART-B is in ```PART-B``` folder

## API Reference

### Creating Model
The command below will create the model with the given params.
```python
    model = ConvNeuralNet(cnn_params, out_features_fc1, dropout, loss, learning_rate, 
                          optimizer["name"], optimizer["default_params"], activation, 
                          epochs, batch_normalisation, init, DEVICE, use_wandb).to(DEVICE)
```

You can find detailed information about format and options for ```cnn_params```, ```out_features_fc1```, ```optimizer```, ```activation``` below.
Note: DEVICE paramter refers to the device being used by

### Training Model
The command below will train the model object for the dataset already passed during creation of ConvNeuralNet object
```python
model.fit()
```

### Getting Predictions
The command below will output label/predicted class for a single instance of test sample passed to the dataset
```python
model.predict(x_test)
```

### cnn_params
```cnn_params``` is a list of dict passed as argument to ```ConvNeuralNet``` model object
#### Example
```python
cnn_params[0] = {
  "in_features": 3,
  "out_features": 32,
  "kernel_size": 2,
  "stride": 1,
  "padding": 1
}
``` 
NOTE: 
 - ```out_features``` of ```cnn_params[i]``` should be equal to ```in_features``` of ```cnn_params[i+1]```
 - ```cnn_params``` can have information about only 5 convolution layers
 - ```cnn_params[0]["in_features"]``` should always be 3 as this is input channel for inaturalist dataset
 - ```stride=1```, ```padding=1```, ```kernel_size=2``` is fixed for now to avoid any errors while training due to excess reduction in image size or to stay under CUDA max memory limit
 

## Available options
### Loss functions
```python
    "cross_entropy": torch.nn.CrossEntropyLoss()
    "mean_squared_error": torch.nn.MSELoss()
```

### Activation functions
```python
   "ReLU": ReLU()
   "SiLU": SiLU()
   "GELU": GELU()
   "Mish": Mish()
```

### Optimizers
```python
    SGD: torch.optim.SGD()
    Adam: torch.optim.Adam()
```

## User Interface
| Commands | Functions |
| --- | --- |
|```python src/train.py``` | fetches parameters passed by command line and trains the model by calling ```main``` function from ```main.py``` file. This further passes arguments to the model to train it. If the model accuracy is above 30%, the model will get saved. 

|```q4_test_data.py``` | fetches the best model saved(best model name should be present in utils.env) and outputs test accuracy and loss. It can be configured to output 10x3 grid of sample predictions of each class.

| ```python src/q4_wandb_sweep``` | contains sweep configuration details. You can change parameters in ```sweep_configuration``` dict to run and log results and configuration details to wandb. Edit ```count``` argument in ```wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=2)``` to change the number of sweeps you wish to execute. NOTE: Bad parameters, like Batch Normalization - False, have not been kept in sweep configurations in order to reduce runs and hence hyperparameter tuning time |


## Arguments Supported

| Name | Default Value | Description |
| --- | --- | --- |
| -wp, --wandb_project |	cs6910-assignment-1 |	Project name used to track experiments in Weights & Biases dashboard |
| -we, --wandb_entity	| me19b110 |	Wandb Entity used to track experiments in the Weights & Biases dashboard |
| -nf, --n_filters |	32 | Number of filters in 1st conv layer |
| -fo, --filter_organisation |	1 | Whether to double/halve/keep same number of filters in subsequent layers |
| -ks, --kernel_size |	2 | Size of kernel/filter for all conv layers (changing this is prone to errors) |
| -st, --stride | 1	| stide for all conv layers (changing this is prone to erros) |
| -pd, --padding | 1	| padding for all conv layers (changing this is prone to erros) |
| -e, --epochs | 10 |	number of epochs |
| -dr, --dropout | 0 | dropout fraction for dense layer only |
| -l, --loss | cross_entropy |	choices: ["mean_squared_error", "cross_entropy"] |
| -o, --optimizer |	SGD |	choices: ["sgd", "Adam"] |
| -lr, --learning_rate |	0.001 |	Learning rate used to optimize model parameters |
| -m, --momentum | 0.5 | Momentum used by SGD optimizer |
| -bt --betas |	(0.9, 0.99) |	Betas used by Adam optimizer |
| -sz, --linear_fc_out_features	| 1024 | Out Features in first dense layer |
| -a, --activation | ReLU | choices: ["ReLU", "SiLU", "GELU", "Mish"] |
| -wb, --use_wandb | 1 | choices: [0, 1]  |
| -bn, --batch_normalisation | 1 | Whether to have batch normalisation or not |

NOTE: Batch normalisation and Data augmentation have been set to True always since they always give better results when used. Data Augmentation has been done while preparing train/test loaders


## Contributors

student name: HARSHIT RAJ  
email: me19b110@smail.iitm.ac.in  
 
course: CS6910 - FUNDAMENTALS OF DEEP LEARNING  
professor: DR. MITESH M. KHAPRA  
 
ta: ASHWANTH KUMAR  
email: cs21m010@smail.iitm.ac.in   
