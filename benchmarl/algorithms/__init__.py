#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm, AlgorithmConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .iddpg import Iddpg, IddpgConfig
from .ippo import Ippo, IppoConfig
from .iql import Iql, IqlConfig
from .isac import Isac, IsacConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig
from .masac import Masac, MasacConfig
from .moma_ac import MomaAC, MomaACConfig
from .momix import MOMix, MOMixConfig
from .pcma import PCMA, PCMAConfig
from .qmix import Qmix, QmixConfig
from .vdn import Vdn, VdnConfig
from .cmomappo import CMOMappo, CMOMappoConfig

classes = [
    "Iddpg",
    "IddpgConfig",
    "Ippo",
    "IppoConfig",
    "Iql",
    "IqlConfig",
    "Isac",
    "IsacConfig",
    "Maddpg",
    "MaddpgConfig",
    "Mappo",
    "MappoConfig",
    "Masac",
    "MasacConfig",
    "MomaAC",
    "MomaACConfig",
    "MOMix",
    "MOMixConfig",
    "PCMA",
    "PCMAConfig",
    "Qmix",
    "QmixConfig",
    "Vdn",
    "VdnConfig",
    "CMOMappo",
    "CMOMappoConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "cmomappo": CMOMappoConfig,
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "maddpg": MaddpgConfig,
    "iddpg": IddpgConfig,
    "masac": MasacConfig,
    "momaac": MomaACConfig,
    "isac": IsacConfig,
    "momix": MOMixConfig,
    "pcma": PCMAConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
    "iql": IqlConfig,
}
