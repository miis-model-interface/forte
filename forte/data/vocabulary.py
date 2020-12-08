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


from collections import defaultdict
from typing import List, Tuple, Any, Iterable


class Vocabulary:
    '''This class is used to save the mapping table from
    tokens, words, labels etc. to indexes.
    There are two methods used for the mapping table.
    One is indexing, which will map the entry to a single interger.
    The other one is one-hot, which will map the entry to an
    one-hot vector.
    '''

    PAD_ID = 0
    PAD_ENTRY = "<PAD>"
    UNK_ENTRY = "<UNK>"

    @staticmethod
    def default_unk():
        return Vocabulary.UNK_ENTRY

    def __init__(self, method: str, use_unk: bool):
        self.element2id_dict = defaultdict()
        self.id2element_dict = defaultdict()
        self.add(self.PAD_ENTRY)

        if use_unk:
            self.add(self.UNK_ENTRY)
            self.id2element_dict.default_factory = Vocabulary.default_unk
            self.element2id_dict.default_factory = Vocabulary.default_unk

        if method not in ("indexing", "one-hot"):
            raise AttributeError("The method %s is not supported in Vocabulary!"
                                 % method)

        self.method = method
        self.use_unk = use_unk

    def __len__(self) -> int:
        return len(self.element2id_dict)

    def get_pad_id(self):
        return self.PAD_ID

    def get_unk_id(self):
        if self.use_unk:
            return self.element2id(self.UNK_ENTRY)
        return None

    def add(self, element: Any):
        if element not in self.element2id_dict:
            idx = len(self)
            self.element2id_dict[element] = idx
            self.id2element_dict[idx] = element

    def get_one_hot(self, idx: int) -> List[int]:
        '''This function will turn '''
        vec = [0] * (len(self)-1)
        vec[idx] = 1
        return vec

    def element2id(self, element: Any) -> int:
        idx = self.element2id_dict[element]
        if self.method == "one-hot":
            return self.get_one_hot(idx)
        return idx

    def id2element(self, idx: int) -> Any:
        return self.id2element_dict[idx]

    def has_key(self, element: Any) -> bool:
        return element in self.element2id_dict

    def items(self) -> Iterable[Tuple[Any, int]]:
        return self.element2id_dict.items()
