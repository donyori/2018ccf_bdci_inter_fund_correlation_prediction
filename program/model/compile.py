from .constants import MAIN_OUTPUT_NAME, AUXILIARY_OUTPUT_NAME
from .metric import tmape, score, t_mae, t_mape, t_tmape, t_score

default_optimizer = 'rmsprop'
default_loss = 'mse'
default_loss_weights = {MAIN_OUTPUT_NAME: 1.0, AUXILIARY_OUTPUT_NAME: 0.2}
default_metrics = ['mae', 'mape', tmape, score, t_mae, t_mape, t_tmape, t_score]
config = {
    'optimizer': default_optimizer,
    'loss': default_loss,
    'loss_weights': default_loss_weights,
    'metrics': default_metrics,
}


def compile_model(model):
    if 'optimizer' not in config:
        config['optimizer'] = default_optimizer
    if 'loss' not in config:
        config['loss'] = default_loss
    if 'loss_weights' not in config:
        config['loss_weights'] = default_loss_weights
    if 'metrics' not in config:
        config['metrics'] = default_metrics
    model.compile(
        optimizer=config['optimizer'],
        loss=config['loss'],
        loss_weights=config['loss_weights'],
        metrics=config['metrics'],
    )
