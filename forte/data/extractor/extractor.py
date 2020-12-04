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

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Union, Iterable, Type
from ft.onto.base_ontology import Annotation, EntityMention
from forte.common.configuration import Config
from forte.data.data_pack import DataPack, DataIndex
from forte.data.extractor.vocabulary import Vocabulary
from forte.data.converter.feature import Feature
from forte.data.extractor.utils import bio_tagging


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
            "vocab_use_pad": True,
            "vocab_use_unk": True,
            "vocab_method": "indexing",
            }
        self.config = Config(config, default_hparams = defaults,
                                    allow_new_hparam = True)

        if self.config.entry_type is None:
            raise AttributeError("Entry_type is needed in the config.")

        if self.config.vocab_method != "raw":
            self.vocab = Vocabulary(method = self.config.vocab_method,
                                    use_pad = self.config.vocab_use_pad,
                                    use_unk = self.config.vocab_use_unk)
        else:
            self.vocab = None

        if self.config.vocab_predefined is not None:
            self.predefined_vocab(self.config.vocab_predefined)


    @property
    def entry_type(self) -> Type[Annotation]:
        return self.config.entry_type

    # Wrapper functions for vocab
    def items(self) -> Iterable:
        if self.vocab:
            return self.vocab.items()
        else:
            return None

    def size(self) -> int:
        if self.vocab:
            return len(self.vocab)
        else:
            return 0

    def add(self, element: Any):
        if self.vocab:
            self.vocab.add(element)
        else:
            pass

    def has_key(self, element: Any):
        if self.vocab:
            return self.vocab.has_key(element)
        else:
            return True

    def id2element(self, idx:int):
        if self.vocab:
            return self.vocab.id2element(idx)
        else:
            return idx

    def element2id(self, element:Any):
        if self.vocab:
            return self.vocab.element2id(element)
        else:
            return element

    def get_dict(self):
        if self.vocab:
            return self.vocab.element2id_dict
        else:
            return None

    def get_pad_id(self)->int:
        if self.vocab:
            return self.vocab.get_pad_id()
        else:
            return 0

    def get_unk_id(self)->int:
        if self.vocab:
            return self.vocab.get_unk_id()
        else:
            return None

    def get_dtype(self):
        if self.vocab:
            return torch.long
        else:
            return str

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

    @abstractmethod
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


class AttributeExtractor(BaseExtractor):
    '''This type of extractor will get the attribute on entry_type
    within one instance.
    '''
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        defaults = {
            "attribute": None,
        }
        self.config = Config(self.config, default_hparams=defaults,
                                            allow_new_hparam=True)
        if self.config.attribute is None:
            raise AttributeError("Attribute is needed for AttributeExtractor.")

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            self.vocab.add(getattr(entry, self.config.attribute))

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        '''The AttributeExtractor only extract one attribute for one entry
        in the instance. There for the output feature will have same number
        of attributes as entries in one instance.
        '''
        data = []
        for entry in pack.get(self.config.entry_type, instance):
            idx = self.element2id(getattr(entry, self.config.attribute))
            data.append(idx)
        # One attribute correspond to one entry, therefore the dim is 1.
        meta_data = {"pad_value": self.get_pad_id(),
                    "dim": 1,
                    "dtype": self.get_dtype()}
        return Feature(data = data, metadata = meta_data,
                        vocab = self.vocab)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        attrs = [self.id2element(x) for x in prediction]
        for entry, attr in zip(pack.get(self.config.entry_type, instance),
                                attrs):
            setattr(entry, self.config.attribute, attr)


class TextExtractor(AttributeExtractor):
    '''A special type of AttributeExtractor, TextExtractor.
    It extract the text attribute on entry within one instance.
    '''
    def __init__(self, config: Union[Dict, Config]):
        config["attribute"] = 'text'
        super().__init__(config)


class CharExtractor(BaseExtractor):
    '''CharExtractor will get each char for each token in the instance.'''
    def __init__(self, config: Union[Dict, Config]):
        defaults = {
                "max_char_length": None
            }
        super().__init__(config)
        self.config = Config(self.config,
                                default_hparams = defaults,
                                allow_new_hparam = True)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for word in pack.get(self.config.entry_type, instance):
            for char in word.text:
                self.add(char)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        data = []
        max_char_length = -1

        for word in pack.get(self.config.entry_type, instance):
            tmp = []
            for char in word.text:
                tmp.append(self.element2id(char))
            data.append(tmp)
            max_char_length = max(max_char_length, len(tmp))

        if self.config.max_char_length is not None:
            max_char_length = min(self.config.max_char_length,
                                    max_char_length)
        # For each token, the output is a list of characters.
        # Therefore the dim is 2.
        meta_data = {"pad_value": self.get_pad_id(),
                    "dim": 2,
                    "dtype": self.get_dtype()}
        return Feature(data = data, metadata = meta_data,
                        vocab = self.vocab)


class BioSeqTaggingExtractor(BaseExtractor):
    '''This class will create a BIO tagging sequence for the attribute of
    the entry_type. E.g. the ner_type of EntityMention in a Sentence. The
    extracted sequence length will be the same as the based_on Annotation.
    E.g. the Token text of a Sentence.
    '''
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        defaults = {
            "attribute": None,
            "strategy": None,
            "based_on": None
        }
        self.config = Config(self.config,
                                default_hparams = defaults,
                                allow_new_hparam = True)

    def bio_variance(self, tag):
        return [(tag, "B"), (tag, "I"), (None, "O")]

    def predefined_vocab(self, predefined: set):
        for tag in predefined:
            for element in self.bio_variance(tag):
                self.add(element)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            attribute = getattr(entry, self.config.attribute)
            for tag_variance in self.bio_variance(attribute):
                self.add(tag_variance)

    def element2id(self, element:Any):
        if self.vocab:
            return super().element2id(element)
        else:
            if element[0]:
                return "-".join(element)
            else:
                return element[1]

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        instance_based_on = list(pack.get(self.config.based_on, instance))
        instance_entry = list(pack.get(self.config.entry_type, instance))
        instance_tagged = bio_tagging(instance_based_on, instance_entry)

        data = []
        for pair in instance_tagged:
            if pair[0] is None:
                new_pair = (None, pair[1])
            else:
                new_pair = (getattr(pair[0], self.config.attribute), pair[1])
            data.append(self.element2id(new_pair))
        meta_data = {"pad_value": self.get_pad_id(),
                    "dim": 1,
                    "dtype": self.get_dtype()}
        return Feature(data = data, metadata = meta_data,
                        vocab = self.vocab)

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        '''This function add the output tag back to the pack. If we
        encounter "I" while its tag is different from the previous tag,
        we will consider this "I" as a "B" and start a new tag here.
        '''
        tags = [self.id2element(x) for x in prediction]
        tag_start = None
        tag_end = None
        tag_type = None
        cnt = 0
        for entry, tag in zip(pack.get(self.config.based_on, instance), tags):
            if tag[1] == "O" or tag[1] == "B":
                if tag_type:
                    entity_mention = EntityMention(pack, tag_start, tag_end)
                    entity_mention.ner_type = tag_type
                tag_start = entry.begin
                tag_end = entry.end
                tag_type = tag[0]
            else:
                if tag[0] == tag_type:
                    tag_end = entry.end
                else:
                    if tag_type:
                        entity_mention = EntityMention(pack, tag_start, tag_end)
                        entity_mention.ner_type = tag_type
                    tag_start = entry.begin
                    tag_end = entry.end
                    tag_type = tag[0]
            cnt += 1

        # Handle the final tag
        if tag_type:
            entity_mention = EntityMention(pack, tag_start, tag_end)
            entity_mention.ner_type = tag_type


class LinkExtractor(BaseExtractor):
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        defaults = {
            "attribute": None,
            "based_on": None
        }
        self.config = Config(self.config,
                                default_hparams = defaults,
                                allow_new_hparam = True)

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            attribute = getattr(entry, self.config.attribute)
            self.add(attribute)

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        instance_based_on = list(pack.get(self.config.based_on, instance))
        instance_entry = list(pack.get(self.config.entry_type, instance))
        parent_entry = [entry.get_parent() for entry in instance_entry]
        child_entry = [entry.get_child() for entry in instance_entry]

        data = [self.element2id(getattr(entry, self.config.attribute))
                    for entry in instance_entry]
        parent_unit_span = []
        child_unit_span = []

        for p, c in zip(parent_entry, child_entry):
            parent_unit_span.append(self.get_index(instance_based_on, p))
            child_unit_span.append(self.get_index(instance_based_on, c))

        meta_data = {
            "parent_unit_span": parent_unit_span,
            "child_unit_span": child_unit_span,
            "pad_value": self.get_pad_id(),
            "dim": 1,
            "dtype": self.get_dtype()
        }

        return Feature(data = data,
                        metadata = meta_data,
                        vocab = self.vocab)

    def get_index(self, inner_entries, span):
        index = DataIndex()
        founds = []
        for i, entry in enumerate(inner_entries):
            if index.in_span(entry, span):
                founds.append(i)
        return [founds[0], founds[-1]+1]
