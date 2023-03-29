import torch.nn as nn
import tensorflow as tf

config = {
    # 'batch_size': 32,
    'val_batch_size': 32,
    'test_batch_size': 64,
    # 'epochs': 50,
    # 'lr': 0.001,
    'beta1': 0.9,
    'beta2': 0.999,
    'epsilon': 1e-08,
    'seed': 42,
    'log_interval': 10,
    'loss_pytorch': nn.CrossEntropyLoss(),
    'loss_keras': tf.keras.losses.CategoricalCrossentropy(),
    'hidden_layers': [10],
    'dropout_fix': 0.2,
    # Sweep config
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64]
        },
        'epochs': {
            'values': [10, 20, 30]
        },
        'lr': {
            'max': 0.01,
            'min': 0.0001
        },
        'dropout': {
            'values': [0.1, 0.2, 0.3]
        },
    },
}

# config.name_pytorch = 'PyTorch L:{} DO:{} LR:{}'.format(str(config.hidden_layers), config.dropout, config.lr)
