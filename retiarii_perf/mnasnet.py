import warnings

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997
_FIRST_DEPTH = 32
_DEFAULT_DEPTHS = [16, 24, 40, 80, 96, 192, 320]
_DEFAULT_CONVOPS = ["dconv", "mconv", "mconv", "mconv", "mconv", "mconv", "mconv"]
_DEFAULT_SKIPS = [False, True, True, True, True, True, True]
_DEFAULT_KERNEL_SIZES = [3, 3, 5, 5, 3, 5, 3]
_DEFAULT_NUM_LAYERS = [1, 3, 3, 3, 2, 4, 1]
_MOBILENET_V2_FILTERS = [16, 24, 32, 64, 96, 160, 320]
_MOBILENET_V2_NUM_LAYERS = [1, 2, 3, 4, 3, 3, 1]


class _ResidualBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x) + x


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, skip, bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = skip and in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack_inverted_residual(in_ch, out_ch, kernel_size, skip, stride, exp_factor, repeats, bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, skip, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, skip, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _stack_normal_conv(in_ch, out_ch, kernel_size, skip, dconv, stride, repeats, bn_momentum):
    assert repeats >= 1
    stack = []
    for i in range(repeats):
        s = stride if i == 0 else 1
        if dconv:
            modules = [
                nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size // 2, stride=s, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            ]
        else:
            modules = [
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, stride=s, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch, momentum=bn_momentum)
            ]
        if skip and in_ch == out_ch and s == 1:
            # use different implementation for skip and noskip to align with pytorch
            stack.append(_ResidualBlock(nn.Sequential(*modules)))
        else:
            stack += modules
        in_ch = out_ch
    return stack


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(depths, alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(self, alpha, depths, convops, kernel_sizes, num_layers,
                 skips, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()
        assert alpha > 0.0
        assert len(depths) == len(convops) == len(kernel_sizes) == len(num_layers) == len(skips) == 7
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths([_FIRST_DEPTH] + depths, alpha)
        exp_ratios = [3, 3, 3, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, depths[0], 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        for conv, prev_depth, depth, ks, skip, stride, repeat, exp_ratio in \
                zip(convops, depths[:-1], depths[1:], kernel_sizes, skips, strides, num_layers, exp_ratios):
            if conv == "mconv":
                # MNASNet blocks: stacks of inverted residuals.
                layers.append(_stack_inverted_residual(prev_depth, depth, ks, skip,
                                                       stride, exp_ratio, repeat, _BN_MOMENTUM))
            else:
                # Normal conv and depth-separated conv
                layers += _stack_normal_conv(prev_depth, depth, ks, skip, conv == "dconv",
                                             stride, repeat, _BN_MOMENTUM)
        layers += [
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True),
                                        nn.Linear(1280, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)


def test_model(model):
    model(torch.randn(2, 3, 224, 224))

#====================Training approach

import sdk
from sdk.mutators.builtin_mutators import ModuleMutator
import datasets

class ModelTrain(sdk.Trainer):
    def __init__(self, device='cuda'):
        super(ModelTrain, self).__init__()
        self.device = torch.device(device)
        self.data_provider = datasets.ImagenetDataProvider(save_path="/data/v-yugzh/imagenet",
                                                    train_batch_size=32,
                                                    test_batch_size=32,
                                                    valid_size=None,
                                                    n_worker=4,
                                                    resize_scale=0.08,
                                                    distort_color='normal')

    def train_dataloader(self):
        return self.data_provider.train

    def val_dataloader(self):
        return self.data_provider.valid

def mnasnet_any(convops, filter_multipliers, kernel_size, num_layer_deltas, skips):
    # Example:
    # convops: ["conv", "mconv", "mconv", "mconv", "mconv", "dconv"]
    # filter_multipliers: [0.75, 1.0, 1.25, 1.0, 1.0, 0.75, 1.25]
    # kernel_size: [3, 5, 3, 3, 3, 5, 3]
    # num_layers: [-1, 0, 1, 0, 0, -1, 1]
    # skips: [False, False, False, True, True, True, False]
    filters = [int(a * b) for a, b in zip(_MOBILENET_V2_FILTERS, filter_multipliers)]
    num_layers = [a + b for a, b in zip(_MOBILENET_V2_NUM_LAYERS, num_layer_deltas)]
    print(num_layers)
    return MNASNet(1.0, filters, convops, kernel_size, num_layers, skips)

def parse_mnasnet_block(line):
    tokens = line.rstrip()[1:-1].split(' ')
    ID_TO_OP = {}
    ID_TO_FILTERS = {}
    ID_TO_KERNEL = {}
    ID_TO_LAYERS = {}
    ID_TO_SKIPS = {}
    for idx, op in enumerate(["conv", "mconv", "dconv"]):
        ID_TO_OP[idx] = op
    for idx, op in enumerate([0.75, 1., 1.25]):
        ID_TO_FILTERS[idx] = op
    for idx, op in enumerate([3, 5]):
        ID_TO_KERNEL[idx] = op
    for idx, op in enumerate([False, True]):
        ID_TO_SKIPS[idx] = op
    print(ID_TO_OP)
    
    block_def = {'convop': [], 'filters': [], 'kernel_size': [], 'num_layers':[], 'skips':[]}
    assert len(tokens) == 7*5
    for i in range(7):
        for idx, op in enumerate([0, 1] + ([-1] if _MOBILENET_V2_NUM_LAYERS[i] > 1 else [0])):
            ID_TO_LAYERS[idx] = op
        block_def['convop'].append(ID_TO_OP[int(tokens[i*5])])
        block_def['num_layers'].append(ID_TO_LAYERS[int(tokens[i*5+1])])
        block_def['filters'].append(ID_TO_FILTERS[int(tokens[i*5+2])])
        block_def['kernel_size'].append(ID_TO_KERNEL[int(tokens[i*5+3])])
        block_def['skips'].append(ID_TO_SKIPS[int(tokens[i*5+4])])
    return block_def

def build_model_from_file(model_type, model_str):
    # with open(model_fp, 'r') as fp:
    #     lines = fp.readlines()
    # model_type = lines[0].rstrip()
    
    if model_type == 'mnasnet':
        block_def = parse_mnasnet_block(model_str)
        return mnasnet_any(block_def['convop'],
                                    block_def['filters'],
                                    block_def['kernel_size'],
                                    block_def['num_layers'],
                                    block_def['skips']
                                    )
    else:
        print('Model not supported')
        return None

#====================Experiment config
def generate(input):
    i, model_str = input
    # mnasnet0_5
    base_model = build_model_from_file('mnasnet', model_str)
                #MNASNet(0.5, _DEFAULT_DEPTHS, _DEFAULT_CONVOPS, _DEFAULT_KERNEL_SIZES,
                #       _DEFAULT_NUM_LAYERS, _DEFAULT_SKIPS)
    exp = sdk.create_experiment('mnist_search', base_model)
    exp.specify_training(ModelTrain)

    exp.specify_mutators([])
    exp.specify_strategy('naive.strategy.main', 'naive.strategy.RandomSampler')
    run_config = {
        'authorName': 'nas',
        'experimentName': 'nas',
        'trialConcurrency': 1,
        'maxExecDuration': '24h',
        'maxTrialNum': 999,
        'trainingServicePlatform': 'local',
        'searchSpacePath': 'empty.json',
        'useAnnotation': False
    } # nni experiment config
    pre_run_config = {
        'name' : f'mnasnet-{i}',
        'x_shape' : [32, 3, 224, 224],
        'x_dtype' : 'torch.float32',
        'y_shape' : [32],
        "y_dtype" : "torch.int64",
        "mask" : False
    }
    exp.run(run_config, pre_run_config = pre_run_config)

import multiprocessing
import os
with open('mnasnet.seq', 'r') as fp:
    seq_lines = fp.readlines()
a_pool = multiprocessing.Pool()
inputs = [ (i, seq_lines[i*2]) for i in range(1000) if not os.path.isfile(f'./generated/mnasnet-{i}.json')]
a_pool.map(generate, inputs)