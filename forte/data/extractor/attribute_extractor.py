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


from typing import Dict, Any, Union
from ft.onto.base_ontology import Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.extractor.base_extractor import BaseExtractor


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
        assert self.config.attribute is not None, \
            "Attribute should not be None."

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            self.add(getattr(entry, self.config.attribute))

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        '''The AttributeExtractor only extract one attribute for one entry
        in the instance. There for the output feature will have same number
        of attributes as entries in one instance.
        '''
        data = []
        for entry in pack.get(self.config.entry_type, instance):
            idx = getattr(entry, self.config.attribute)
            if self.vocab:
                idx = self.element2id(idx)
            data.append(idx)
        # Data only has one dimension, therefore dim is 1.
        meta_data = {"pad_value": self.get_pad_id(),
                    "dim": 1,
                    "dtype": int if self.vocab else str}
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