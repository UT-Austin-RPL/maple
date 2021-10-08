"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from maple.torch.networks.basic import (
    Clamp, ConcatTuple, Detach, Flatten, FlattenEach, Split, Reshape,
)
from maple.torch.networks.cnn import BasicCNN, CNN, MergedCNN, CNNPolicy
from maple.torch.networks.dcnn import DCNN, TwoHeadDCNN
from maple.torch.networks.feat_point_mlp import FeatPointMlp
from maple.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from maple.torch.networks.linear_transform import LinearTransform
from maple.torch.networks.normalization import LayerNorm
from maple.torch.networks.mlp import (
    Mlp, ConcatMlp, MlpPolicy, TanhMlpPolicy,
    MlpQf,
    MlpQfWithObsProcessor,
    ConcatMultiHeadedMlp,
)
from maple.torch.networks.pretrained_cnn import PretrainedCNN
from maple.torch.networks.two_headed_mlp import TwoHeadMlp

__all__ = [
    'Clamp',
    'ConcatMlp',
    'ConcatMultiHeadedMlp',
    'ConcatTuple',
    'BasicCNN',
    'CNN',
    'CNNPolicy',
    'DCNN',
    'Detach',
    'FeatPointMlp',
    'Flatten',
    'FlattenEach',
    'LayerNorm',
    'LinearTransform',
    'ImageStatePolicy',
    'ImageStateQ',
    'MergedCNN',
    'Mlp',
    'PretrainedCNN',
    'Reshape',
    'Split',
    'TwoHeadDCNN',
    'TwoHeadMlp',
]

