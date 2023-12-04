# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch

import torch.nn.functional as F
from model_separable_rnnt import label_collate


class ScriptGreedyDecoder(torch.nn.Module):
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    def __init__(self, blank_index, model, max_symbols_per_step=30):
        super().__init__()
        # assert isinstance(model, torch.jit.ScriptModule)
        # assert not model.training
        self.eval()
        self._model = model
        self._blank_id = blank_index
        self._SOS = -1
        assert max_symbols_per_step > 0
        self._max_symbols_per_step = max_symbols_per_step

    @torch.jit.export
    def forward(self, x: torch.Tensor, out_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        # Apply optional preprocessing
        logits, logits_lens = self._model.encoder(x, out_lens)

        output: List[List[int]] = []
        for batch_idx in range(logits.size(0)):
            inseq = logits[batch_idx, :, :].unsqueeze(1)
            # inseq: TxBxF
            logitlen = logits_lens[batch_idx]
            sentence = self._greedy_decode(inseq, logitlen)
            output.append(sentence)

        return logits, logits_lens, output

    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor) -> List[int]:
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        label: List[int] = []
        for time_idx in range(int(out_len.item())):
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and symbols_added < self._max_symbols_per_step:
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden
                )
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        return label

    def _pred_step(self, label: int, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if label == self._SOS:
            return self._model.prediction(None, hidden)
        if label > self._blank_id:
            label -= 1
        label = torch.tensor([[label]], dtype=torch.int64)
        return self._model.prediction(label, hidden)

    def _joint_step(self, enc: torch.Tensor, pred: torch.Tensor, log_normalize: bool=False) -> torch.Tensor:
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels: List[int]) -> int:
        return self._SOS if len(labels) == 0 else labels[-1]
