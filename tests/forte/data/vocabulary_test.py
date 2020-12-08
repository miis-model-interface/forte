#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest
from forte.data.vocabulary import Vocabulary
from itertools import product


class VocabularyTest(unittest.TestCase):
    def argmax(self, one_hot):
        idx = -1
        for i, flag in enumerate(one_hot):
            if flag:
                self.assertTrue(idx == -1)
                idx = i
        self.assertTrue(idx != -1)
        return idx

    def test_indexing_vocab(self):
        for use_unk in [True, False]:
            vocab = Vocabulary(method="indexing", use_unk=use_unk)
            
            # Check PAD_ELEMENT
            expected_pad_repr = 0
            self.assertEqual(expected_pad_repr, vocab.get_pad_repr())
            
            # Check UNK_ELEMENT
            if use_unk:
                expected_unk_repr = 1
                self.assertEqual(expected_unk_repr,
                            vocab.element2repr(Vocabulary.UNK_ELEMENT))
            
            # Check vocabulary add, element2repr and id2element
            elements = ["EU", "rejects", "German", "call", 
                        "to", "boycott", "British", "lamb", "."]

            for ele in elements:
                vocab.add(ele)

            reprs = [vocab.element2repr(ele) for ele in elements]
            
            self.assertTrue(len(reprs)>0)
            self.assertTrue(isinstance(reprs[0], int))

            recovered_elements = []
            for rep in reprs:
                idx = rep
                recovered_elements.append(vocab.id2element(idx))

            self.assertListEqual(elements, recovered_elements)


    def test(self):
        flags = [True, False]
        for use_unk, method in product(flags, ["indexing", "one-hot"]):
            vocab = Vocabulary(method=method, use_unk=use_unk)
            
            
            
            if use_unk:
                self.assertEqual(vocab.get_unk_id(), vocab.element2repr(Vocabulary.UNK_ELEMENT))

            sentence = "EU rejects German call to boycott British lamb ."
            tokens = sentence.split(" ")

            for tok in tokens:
                vocab.add(tok)

            self.assertEqual(len(set(tokens)) + int(use_unk), len(vocab))

            ids = [vocab.element2repr(tok) for tok in tokens]
            if method == "indexing":
                self.assertTrue(isinstance(ids[0], int))
            else:
                print(ids[0])
                self.assertTrue(isinstance(ids[0], list))
                ids = [self.argmax(one_hot) for one_hot in ids]
            print(ids)
            reverted = [vocab.id2element(idx) for idx in ids]
            self.assertListEqual(tokens, reverted)


if __name__ == '__main__':
    unittest.main()
