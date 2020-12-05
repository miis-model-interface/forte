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
from typing import Dict, Any, Union, Iterable, Type
from ft.onto.base_ontology import Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.extractor.vocabulary import Vocabulary
from forte.data.converter.feature import Feature


class BaseExtractor(ABC):
    '''This class is used to get feature from the datapack and also
    add prediction back to datapack.
    '''
    def __init__(self, config: Union[Dict, Config]):
        '''Config will need to contains some value to initialize the
        extractor.
        Entry_type: Type[EntryType], every extractor will get feature by loop on
            one type of entry in the instance. e.g. Token, EntityMention.
        Vocab_use_pad, Vocab_use_unk, Vocab_method" are used to configurate the
                vocabulary class.
        Vocab_predefined: a set of elements be added to the vocabulary.
        '''
        defaults = {
            "entry_type": None,
            "vocab_method": "indexing",
            "vocab_use_unk": True,
            }
        self.config = Config(config,
                            default_hparams = defaults,
                            allow_new_hparam = True)

        assert self.config.entry_type is not None, \
            "Entry_type should not be None."

        if self.config.vocab_method != "raw":
            self.vocab = Vocabulary(method = self.config.vocab_method,
                                    use_unk = self.config.vocab_use_unk)
        else:
            self.vocab = None

    @property
    def entry_type(self) -> Type[Annotation]:
        return self.config.entry_type

    def items(self) -> Iterable:
        assert self.vocab, \
            "Items should not be called, when vocab is None."
        return self.vocab.items()

    def size(self) -> int:
        assert self.vocab, \
            "Items should not be called, when vocab is None."
        return len(self.vocab)

    def add(self, element: Any):
        assert self.vocab, \
            "Items should not be called, when vocab is None."
        self.vocab.add(element)

    def has_key(self, element: Any) -> bool:
        assert self.vocab, \
            "Items should not be called, when vocab is None."
        return self.vocab.has_key(element)

    def id2element(self, idx:int):
        '''For raw vocabulary, map id to itself.
        '''
        assert self.vocab, \
            "Items should not be called, when vocab is None."
        return self.vocab.id2element(idx)

    def element2id(self, element:Any):
        assert self.vocab, \
            "Items should not be called, when vocab is None."
        return self.vocab.element2id(element)

    def get_dict(self):
        assert self.vocab, \
            "Items should not be called, when vocab is None."
        return self.vocab.element2id_dict

    def get_pad_id(self)->int:
        if self.vocab:
            return self.vocab.get_pad_id()
        else:
            return Vocabulary.PAD_ID

    def predefined_vocab(self, predefined: set):
        '''This function will add elements from the passed-in predefined
        set to the vocab. Different extractor might have different strategies
        to add these elements.
        '''
        for element in predefined:
            self.add(element)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        '''This function is used when user want to add element to vocabulary
        using the current instance. e.g. add all tokens in one sentence to
        the vocabulary.
        '''
        raise NotImplementedError()

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        '''This function will extract feature from one instance in the pack.
        '''
        raise NotImplementedError()

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        '''This function will add prediction to the pack according to different
        type of extractor.
        '''
        raise NotImplementedError()
