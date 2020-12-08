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
from typing import List, Tuple, Union, Any, Iterable

class Vocabulary:
    '''This class maps element to representation. Element
    could be any hashable type and there are two types of
    representations, namely, "indexing" and "one-hot". There
    are two special types of element, namely PAD_ELEMENT
    and UNK_ELEMENT.
    For "indexing" vocabulary,
        Element:  <PAD>  ele1   ele2   ele3  ...
        Id:         0      1      2      3   ...
        Repr:       0      1      2      3   ...
    For "one-hot" vocabulary,
        Element:  <PAD>  ele1   ele2   ele3  ...
        id:        -1      0      1      2   ...
        Repr:      [0,    [1,    [0,    [0,  ...
                    0,     0,     1,     0,  ...
                    0,     0,     0,     1,  ...
                    0,     0,     0,     0,  ...
                    ...]   ...]   ...]   ...]
    If vocabulary uses UNK_ELEMENT, the first element,
    "ele1" will be UNK_ELEMENT and any other elements
    that cannot be found in the current vocabulary will
    be mapped to the UNK_ELEMENT. Otherwise, UNK_ELEMENT
    is not used. Error will occur when querying unknown
    element in the vocabulary.
    '''
    PAD_ELEMENT = "<PAD>"
    UNK_ELEMENT = "<UNK>"

    def __init__(self, method: str, use_unk: bool):
        self.element2id_dict = defaultdict()
        self.id2element_dict = defaultdict()

        if method == "indexing":
            self.next_id = 0
        elif method == "one-hot":
            self.next_id = -1
        else:
            raise AttributeError("The method %s \
                is not supported in Vocabulary!" % method)

        self.add(Vocabulary.PAD_ELEMENT)

        if use_unk:
            self.add(Vocabulary.UNK_ELEMENT)
            self.element2id_dict.default_factory = \
                self.get_unk_id

        self.method = method
        self.use_unk = use_unk

    def add(self, element: Any):
        self.element2id_dict[element] = self.next_id
        self.id2element_dict[self.next_id] = element
        self.next_id += 1

    def id2repr(self, idx: int) -> List[int]:
        if self.method == "indexing":
            return idx
        vec = [0]*len(self)
        if idx == -1:
            return vec
        else:
            vec[idx] = 1
            return vec

    def get_unk_id(self) -> int:
        return self.element2id_dict[Vocabulary.UNK_ELEMENT]

    def get_pad_repr(self) -> Union[int, List[int]]:
        return self.id2repr(
            self.element2id_dict[Vocabulary.PAD_ELEMENT])

    def element2repr(self, element: Any) -> Union[int, List[int]]:
        return self.id2repr(self.element2id_dict[element])

    def id2element(self, idx: int) -> Any:
        return self.id2element_dict[idx]

    def __len__(self) -> int:
        return len(self.element2id_dict)

    def has_element(self, element: Any) -> bool:
        return element in self.element2id_dict

    def items(self) -> Iterable[Tuple[Any, int]]:
        return self.element2id_dict.items()
