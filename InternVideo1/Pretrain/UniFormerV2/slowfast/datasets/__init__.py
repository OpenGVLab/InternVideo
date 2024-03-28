#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import DATASET_REGISTRY, build_dataset  # noqa
from .charades import Charades  # noqa
from .imagenet import Imagenet  # noqa
from .kinetics import Kinetics  # for Kinetics (dense sampling)
from .kinetics_sparse import Kinetics_sparse  # for Kinetics (sparse sampling)
from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
from .sth import Sth  # shared for Something-Something V1 and V2
from .anet import Anet # shared for ActivityNet and HACS
from .mit import Mit # for Moments in Time
