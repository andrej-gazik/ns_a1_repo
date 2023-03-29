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
    # 'hidden_layers': [10],
    'hidden_layers': [[0], [10], [16, 8], [10,5], [17,14,11,8], [18,12,8,6], [18,16,14,12,10,8,6], [19,18,17,16,15,14,13,12,11,10,9,8,7,6,5]],
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
            'values': [16, 32, 64, 128]
        },
        'epochs': {
            'values': [10, 20, 30, 40, 50, 70, 100]
        },
        'lr': {
            'max': 0.01,
            'min': 0.0001
        },
        'dropout': {
            'max': 0.8,
            'min': 0.2
        },
    },
}

# config.name_pytorch = 'PyTorch L:{} DO:{} LR:{}'.format(str(config.hidden_layers), config.dropout, config.lr)
