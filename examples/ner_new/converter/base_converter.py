# Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import List
import torch
from forte.common.resources import Resources
from forte.common.configuration import Config

class Converter():
    """ Base class.
    """
    def __init__():
        pass

    @abstractmethod
    def initialize(self, resources: Resources, configs: Config):
        # TODO: self.vocab = resource["vocab"]
        pass
    
    @abstractmethod
    def pack2tensor(self, instance:List)->torch.Tensor:
        pass

    @abstractmethod
    def tensor2pack(self, tensor:torch.Tensor)->List:
        pass

    