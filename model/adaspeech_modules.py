from matplotlib.pyplot import sca
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from collections import OrderedDict

class LayerNorm(torch.nn.Module):
    def __init__(self, nout: int):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(nout, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x.transpose(1, -1))
        x = x.transpose(1, -1)
        
        return x

class UtteranceEncoder(nn.Module):
    """ Acoustic modeling """
    
    def __init__(self, model_config):
        super(UtteranceEncoder, self).__init__()
        self.idim = model_config["UtteranceEncoder"]["idim"]
        self.n_layers = model_config["UtteranceEncoder"]["n_layers"]
        self.n_chans = model_config["UtteranceEncoder"]["n_chans"]
        self.kernel_size = model_config["UtteranceEncoder"]["kernel_size"]
        self.pool_kernel = model_config["UtteranceEncoder"]["pool_kernel"]
        self.dropout_rate = model_config["UtteranceEncoder"]["dropout_rate"]
        self.stride = model_config["UtteranceEncoder"]["stride"]
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                    nn.Conv1d(
                        self.idim,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                    nn.Conv1d(
                        self.n_chans,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        
    def forward(self, xs):
        xs = self.conv(xs)
        xs = F.avg_pool1d(xs, xs.size(-1))

        return xs

class PhonemeLevelEncoder(nn.Module):
    """ Phoneme level encoder """

    def __init__(self, model_config):
        super(PhonemeLevelEncoder, self).__init__()
        self.idim = model_config["PhonemeLevelEncoder"]["idim"]
        self.n_layers = model_config["PhonemeLevelEncoder"]["n_layers"]
        self.n_chans = model_config["PhonemeLevelEncoder"]["n_chans"]
        self.kernel_size = model_config["PhonemeLevelEncoder"]["kernel_size"]
        self.dropout_rate = model_config["PhonemeLevelEncoder"]["dropout_rate"]
        self.stride = model_config["PhonemeLevelEncoder"]["stride"]
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                    nn.Conv1d(
                        self.idim,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                    nn.Conv1d(
                        self.n_chans,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        self.linear = nn.Linear(self.n_chans, 4)

    def forward(self, xs):
        xs = self.conv(xs)
        xs = self.linear(xs.transpose(1,2))

        return xs

class PhonemeLevelPredictor(nn.Module):
    """ Phoneme Level Predictor """

    def __init__(self, model_config):
        super(PhonemeLevelPredictor, self).__init__()
        self.idim = model_config["PhonemeLevelPredictor"]["idim"]
        self.n_layers = model_config["PhonemeLevelPredictor"]["n_layers"]
        self.n_chans = model_config["PhonemeLevelPredictor"]["n_chans"]
        self.kernel_size = model_config["PhonemeLevelPredictor"]["kernel_size"]
        self.dropout_rate = model_config["PhonemeLevelPredictor"]["dropout_rate"]
        self.stride = model_config["PhonemeLevelPredictor"]["stride"]
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ("conv1d_1",
                    nn.Conv1d(
                        self.idim,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", LayerNorm(self.n_chans)),
                    ("dropout_1", nn.Dropout(self.dropout_rate)),
                    ("conv1d_2",
                    nn.Conv1d(
                        self.n_chans,
                        self.n_chans,
                        self.kernel_size,
                        stride = self.stride,
                        padding = (self.kernel_size - 1) // 2,
                    )
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", LayerNorm(self.n_chans)),
                    ("dropout_2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )
        self.linear = nn.Linear(self.n_chans, 4)

    def forward(self, xs):
        xs = self.conv(xs)
        xs = self.linear(xs.transpose(1,2))

        return xs

class Condional_LayerNorm(nn.Module):
    """Conditional Layer Normalization """

    def __init__(self, normal_shape, model_config, epsilon=1e-5):
        super(Condional_LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = model_config["transformer"]["encoder_hidden"]
        self.epsilon = model_config["ConditionalLayerNorm"]["epsilon"]
        self.W_scale = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.W_bias = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)

        return y
