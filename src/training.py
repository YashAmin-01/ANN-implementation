import os.path

from utils.common import read_config
from utils.data_mgmt import get_data
from utils.model import create_model, save_model
import argparse


def training(config_path):
    config = read_config(config_path)

    validation_datasize = config['params']['validation_datasize']
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config['params']['loss_function']
    OPTIMIZER = config['params']['optimizer']
    METRICS = config['params']['metrics']
    NUM_CLASSES = config['params']['num_classes']

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config['params']['epochs']
    VALIDATION = (X_valid, y_valid)

    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=VALIDATION)

    model_name = config['artifacts']['model_name']
    artifacts_dir = config['artifacts']['artifacts_dir']
    model_dir = config['artifacts']['model_dir']
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    save_model(model, model_name, model_dir_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--config', '-c', default='config.yaml')
    args.add_argument('--secret', '-s', default='secrets.yaml')

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)