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


from abc import ABC
from typing import Dict, Any, Union, Type, Callable
from ft.onto.base_ontology import Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.vocabulary import Vocabulary
from forte.data.converter.feature import Feature


class BaseExtractor(ABC):
    '''This class is used to get Feature from Datapack and
    add prediction back to Datapack.
    '''
    def __init__(self, config: Union[Dict, Config]):
        '''Config: {"entry_type" : required, Type[Annotation],
                    "vocab_method": optional, str,
                        "raw", "indexing", "one-hot" are supported,
                        default is "indexing",
                    "vocab_use_unk": optional, bool,
                        default is True}
        '''
        defaults = {
            "vocab_method": "indexing",
            "vocab_use_unk": True,
        }

        self.config = Config(config, defaults, allow_new_hparam=True)

        assert hasattr(self.config, "entry_type"), \
            "Entry_type should not be None."

        if self.config.vocab_method != "raw":
            self.vocab = Vocabulary(method=self.config.vocab_method,
                                    use_unk=self.config.vocab_use_unk)
        else:
            self.vocab = None

        self.items = self.wrap_vocab_fn("items")
        self.size = self.wrap_vocab_fn("__len__")
        self.add = self.wrap_vocab_fn("add")
        self.has_key = self.wrap_vocab_fn("has_key")
        self.id2element = self.wrap_vocab_fn("id2element")
        self.element2repr = self.wrap_vocab_fn("element2repr")
        self.get_dict = self.wrap_vocab_fn("get_dict")

    @property
    def entry_type(self) -> Type[Annotation]:
        return self.config.entry_type

    @property
    def vocab_method(self) -> str:
        return self.config.vocab_method

    def wrap_vocab_fn(self, func_name: str) -> Callable:
        def wrapper(*args, **kwargs):
            assert self.vocab, """When vocab_mehtod is raw,
            vocabulary is not built and operation on vocabulary should not
            be called."""
            return getattr(self.vocab, func_name)(*args, **kwargs)
        return wrapper

    def get_pad_id(self) -> int:
        '''PAD ID is always 0.'''
        return 0

    def predefined_vocab(self, predefined: set):
        '''This function will add elements from the
        passed-in predefined set to the vocab. Different
        extractors might have different strategies to add
        these elements. Override this function if necessary.
        '''
        for element in predefined:
            self.add(element)

    def update_vocab(self, pack: DataPack,
                    instance: Annotation):
        '''This function will extract the feature from
        instance and add element in the feature to vocabulary.
        '''
        raise NotImplementedError()

    def extract(self, pack: DataPack,
                instance: Annotation) -> Feature:
        '''This function will extract feature from
        one instance in the pack.
        '''
        raise NotImplementedError()

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        '''This function will add prediction to the pack.
        '''
        raise NotImplementedError()
