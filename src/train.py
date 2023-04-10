import argparse
from utils.prepare_params import get_cnn_params
import wandb
from params.default_params import default_credentials, default_model_params, optimizer_param_map, default_use_wandb
from params.parser_params import parser_choices
from main import main


parser = argparse.ArgumentParser(
                    prog='train',
                    description='Suppy parameters to convolution neural network to run and log results in wandb.ai',
                    epilog="That's all")

optimizer = optimizer_param_map[default_model_params["optimizer"]]

parser.add_argument("-wp", "--wandb_project", choices= parser_choices["wandb_project"], default=default_credentials["wandb_project"])
parser.add_argument("-we", "--wandb_entity", choices=parser_choices["wandb_entity"], default=default_credentials["wandb_entity"])
parser.add_argument("-nf", "--n_filters", default=default_model_params["n_filters"], type=int)
parser.add_argument("-fo", "--filter_organisation", choices=parser_choices["filter_organisation"], default=default_model_params["filter_organisation"], type=int)
parser.add_argument("-ks", "--kernel_size", default=default_model_params["kernel_size"], type=int)
parser.add_argument("-st", "--stride", default=default_model_params["stride"], type=int)
parser.add_argument("-pd", "--padding", default=default_model_params["padding"], type=int)
parser.add_argument("-e", "--epochs", default=default_model_params["epochs"], type=int)
parser.add_argument("-dr", "--dropout", default=default_model_params["dropout"])
parser.add_argument("-l", "--loss", choices= parser_choices["loss"], default=default_model_params["loss"])
parser.add_argument("-o", "--optimizer", choices=parser_choices["optimizer"], default=default_model_params["optimizer"])
parser.add_argument("-lr", "--learning_rate", default=optimizer["default_params"]["learning_rate"], type=float)
parser.add_argument("-m", "--momentum", default=optimizer_param_map["momentum"]["default_params"]["gamma"], type=float)
parser.add_argument("-beta", "--beta", default=optimizer_param_map["rmsprop"]["default_params"]["beta"], type=float)
parser.add_argument("-beta1", "--beta1", default=optimizer_param_map["nadam"]["default_params"]["beta1"], type=float)
parser.add_argument("-beta2", "--beta2", default=optimizer_param_map["nadam"]["default_params"]["beta2"], type=float)
parser.add_argument("-eps", "--epsilon", default=optimizer_param_map["nadam"]["default_params"]["epsilon"], type=float)
parser.add_argument("-sz", "--linear_fc_out_features", default=default_model_params["linear_fc_out_features"], type=int)
parser.add_argument("-a", "--activation", choices=parser_choices["activation"], default=default_model_params["activation"])
parser.add_argument("-wb", "--use_wandb", choices=[0, 1], default=default_use_wandb, type=int)
args = parser.parse_args()

optimizer = optimizer_param_map[args.optimizer]
for key in optimizer["default_params"].keys():
    optimizer["default_params"][str(key)] = getattr(args, str(key))

print(args)
epochs = args.epochs
linear_fc_out_features= args.linear_fc_out_features
activation = args.activation
use_wandb = args.use_wandb
n_filters = args.n_filters
filter_organisation = args.filter_organisation
dropout = args.dropout
loss = args.loss
learning_rate = args.learning_rate
kernel_size = args.kernel_size
stride = args.stride
padding = args.padding

if use_wandb:
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    run.name = "ac_{}_opt_{}".format(args.activation, args.optimizer)
    wandb.log({
        epochs: epochs,
        linear_fc_out_features: linear_fc_out_features,
        activation: activation,
        use_wandb: use_wandb,
        n_filters: n_filters,
        filter_organisation: filter_organisation,
        dropout: dropout,
        loss: loss,
        learning_rate: learning_rate,
        optimizer: optimizer["name"],
        kernel_size: kernel_size,
        stride : stride,
        padding: padding
    })
    run.log_code()

if __name__ == '__main__':
    cnn_params = get_cnn_params(n_filters, filter_organisation, kernel_size, stride, padding)
    main(epochs, activation, cnn_params, linear_fc_out_features, dropout, loss, learning_rate, optimizer, use_wandb)