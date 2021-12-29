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
import time

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
        #assert isinstance(model, torch.jit.ScriptModule)
        # assert not model.training
        self.eval()
        self._model = model
        self._blank_id = blank_index
        self._SOS = -1
        assert max_symbols_per_step > 0
        self._max_symbols_per_step = max_symbols_per_step

    @torch.jit.export
    def forward_dec_single_batch(self, logits: torch.Tensor, logits_lens: torch.Tensor, int8, bf16) -> List[List[int]]:
        """Returns a list of sentences given an input batch.

        Args:
            logits: logits produced by encoder
            logits_lens: length of each logits

        Returns:
            list containing batch number of sentences (strings).
        """
        import intel_pytorch_extension as ipex
        logits = logits.to(ipex.DEVICE)
        if int8:
            if bf16:
                # enable bf16 for decoder part
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        else:
            # the case of int8 = False and bf16 = True had already processed in higher level
            pass

        # inseq: TxBxF
        logitlen = logits_lens[0]
        sentence = self._greedy_decode(logits, logitlen)

        return [sentence]

    @torch.jit.export
    def forward_single_batch(self, x: torch.Tensor, out_lens: torch.Tensor, conf, int8, bf16, run_mode="inference") -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
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

        t0 = time.time()
        if int8:
            import intel_pytorch_extension as ipex
            with ipex.AutoMixPrecision(conf, running_mode=run_mode):
                logits, logits_lens = self._model.encoder(x, out_lens)

            # TODO: support directly reorder data from int8 to bf16
            # This is an workaround here to transfer logits to cpu
            # to reorder data from int8 to fp32
            logits = logits.to("cpu")
            logits = logits.to(ipex.DEVICE)

            if bf16:
                # enable bf16 for decoder part
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        else:
            # the case of int8 = False and bf16 = True had already processed in higher level
            logits, logits_lens = self._model.encoder(x, out_lens)

        #os.environ['OMP_NUM_THREADS'] = '1'
        t1 = time.time()
        # inseq: TxBxF
        logitlen = logits_lens[0]
        sentence = self._greedy_decode(logits, logitlen)
        t2 = time.time()

        return logits, logits_lens, [sentence], t1-t0, t2-t1

    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor) -> List[int]:
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        label: List[int] = []
        timesteps = int(out_len.item())
        last_symb = self._SOS
        time_idx = 0
        x.unsqueeze_(0)

        symb_added = 0
        while 1:
            g, hidden_prime = self._pred_step(last_symb, hidden)
            logp = self._joint_step_nolog(x[:, :, time_idx, :], g)

            # get index k, of max prob
            _, k = logp.max(0)
            k = k.item()

            if k == self._blank_id or symb_added >= self._max_symbols_per_step:
                time_idx += 1
                if time_idx >= timesteps:
                    break
                symb_added = 0
            else:
                last_symb = k
                label.append(k)
                symb_added += 1
                hidden = hidden_prime

        return label

    """
    def _greedy_decode_origin(self, x: torch.Tensor, out_len: torch.Tensor) -> List[int]:
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        label: List[int] = []
        for time_idx in range(int(out_len.item())):
            f = x[:, time_idx, :].unsqueeze_(0)

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
    """

    def _pred_step(self, label: int, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        #if label > self._blank_id:
        #    label -= 1
        label = torch.tensor([[label]], dtype=torch.int64)
        result = self._model.prediction(label, hidden)
        return result

    def _joint_step_nolog(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return self._model.joint(enc, pred)[0, 0, 0, :]

    def _joint_step(self, enc: torch.Tensor, pred: torch.Tensor, log_normalize: bool=False) -> torch.Tensor:
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels: List[int]) -> int:
        return self._SOS if len(labels) == 0 else labels[-1]

    @torch.jit.export
    def forward_enc_batch(self, x: torch.Tensor, out_lens: torch.Tensor, conf, int8, run_mode="inference") -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            logits and logits lens
        """
        # Apply optional preprocessing
        # int8 encoder + bf16 decoder
        if int8:
            import intel_pytorch_extension as ipex
            with ipex.AutoMixPrecision(conf, running_mode=run_mode):
                logits, logits_lens = self._model.encoder(x, out_lens)

            # TODO: support directly reorder data from int8 to bf16
            # This is an workaround here to transfer logits to cpu
            # to reorder data from int8 to fp32
            logits = logits.to("cpu")
        else:
            # the case of int8 = False and bf16 = True had already processed in higher level
            logits, logits_lens = self._model.encoder(x, out_lens)

        return logits, logits_lens

    @torch.jit.export
    def forward_dec_batch(self, logits: torch.Tensor, logits_lens: torch.Tensor, int8, bf16) -> Tuple[List[List[int]], float]:
        """Returns a list of sentences given an input batch.

        Args:
            logits, logits_lens: encoder input

        Returns:
            list containing batch number of sentences (strings).
        """
        # Apply optional preprocessing
        # int8 encoder + bf16 decoder
        import intel_pytorch_extension as ipex
        logits = logits.to(ipex.DEVICE)
        if int8:
            if bf16:
                # enable bf16 for decoder part
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        else:
            # the case of int8 = False and bf16 = True had already processed in higher level
            pass

        sentences = self._greedy_decode_batch(logits, logits_lens)

        return sentences

    @torch.jit.export
    def forward_batch(self, x: torch.Tensor, out_lens: torch.Tensor, conf, int8, bf16, run_mode="inference") -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        """
        # Apply optional preprocessing
        # int8 encoder + bf16 decoder
        t0 = time.time()
        if int8:
            import intel_pytorch_extension as ipex
            with ipex.AutoMixPrecision(conf, running_mode=run_mode):
                logits, logits_lens = self._model.encoder(x, out_lens)

            # TODO: support directly reorder data from int8 to bf16
            # This is an workaround here to transfer logits to cpu
            # to reorder data from int8 to fp32
            logits = logits.to("cpu")
            logits = logits.to(ipex.DEVICE)

            if bf16:
                # enable bf16 for decoder part
                ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)
        else:
            # the case of int8 = False and bf16 = True had already processed in higher level
            logits, logits_lens = self._model.encoder(x, out_lens)

        t1 = time.time()
        sentences = self._greedy_decode_batch(logits, logits_lens)
        t2 = time.time()

        return logits, logits_lens, sentences, t1-t0, t2-t1
        """
        t0 = time.time()
        logits, logits_lens = self.forward_enc_batch(x, out_lens, conf, int8, bf16, run_mode)
        t1 = time.time()
        sentences = self.forward_dec_batch(logits, logits_lens, int8, bf16)
        t2 = time.time()
        return logits, logits_lens, sentences, t1-t0, t2-t1

    def count_nonzero(self, x: torch.Tensor) -> int:
        return x.nonzero().shape[0]

    def _greedy_decode_batch(self, x: torch.Tensor, out_lens: torch.Tensor) -> List[List[int]]:
        batch_size = x.size(0)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        max_len = out_lens.max().item()
        max_lens = torch.tensor([max_len-1] * batch_size, dtype=torch.int64)
        # pos 0 of label_tensor is set to _SOS to simplify computation
        # real label start from pos 1
        label_tensor = torch.tensor([self._SOS]).repeat(batch_size, max_len*self._max_symbols_per_step)  # (B, T/2*max_symbols_per_step)
        # (row, col) of current labels end
        label_row = torch.tensor(list(range(batch_size)))
        label_col = torch.tensor([0] * batch_size)
        # this list will be used to return labels to caller
        label_copy = [0] * batch_size
        # initially time_idx is 0 for all input
        # then advance time_idx for each 'track' when needed and update f
        f = x[:, 0, :].unsqueeze(1)
        time_idxs = torch.tensor([0] * batch_size, dtype=torch.int64)

        not_blank = True
        blank_vec = torch.tensor([0] * batch_size, dtype=torch.int)
        symbols_added = torch.tensor([0] * batch_size, dtype=torch.int)

        while True:
            g, hidden_prime = self._pred_step_batch(
                label_tensor.gather(1, label_col.unsqueeze(1)),
                hidden,
                batch_size
            )
            logp = self._joint_step_batch(f, g, log_normalize=False)

            # get index k, of max prob
            v, k = logp.max(1)

            # if any of the output is blank, pull in the next time_idx for next f
            # tmp_blank_vec is the vect used to mix new hidden state with previous hidden state
            # blank_vec is the baseline of blank_vec, it turns to blank only when run out of time_idx
            blankness = k.eq(self._blank_id)
            time_idxs = time_idxs + blankness
            symbols_added *= blankness.logical_not()
            # it doesn't matter if blank_vec is update now or later,
            # tmp_blank_vec always get correct value for this round
            blank_vec = time_idxs.ge(out_lens)
            tmp_blank_vec = blank_vec.logical_or(blankness)

            if self.count_nonzero(blank_vec) == batch_size:
                # all time_idxs processed, stop
                break
            else:
                # If for sample blankid already encountered, then stop
                # update hidden values until input from next time step.
                # So we would mix value of hidden and hidden_prime together,
                # keep values in hidden where blank_vec[i] is true
                if hidden == None:
                    hidden = [torch.zeros_like(hidden_prime[0]), torch.zeros_like(hidden_prime[1])]

                idx = (tmp_blank_vec.eq(0)).nonzero(as_tuple=True)[0]
                hidden[0][:, idx, :] = hidden_prime[0][:, idx, :]
                hidden[1][:, idx, :] = hidden_prime[1][:, idx, :]

            label_col += tmp_blank_vec.eq(False)
            label_tensor.index_put_([label_row, label_col], (k-self._SOS)*tmp_blank_vec.eq(False), accumulate=True)

            symbols_added += tmp_blank_vec.eq(False)
            sym_ge_vec = symbols_added.ge(self._max_symbols_per_step)
            if sym_ge_vec.count_nonzero() != 0:
                time_idxs += sym_ge_vec
                blankness.logical_or(sym_ge_vec)
                symbols_added *= symbols_added.lt(self._max_symbols_per_step)

            # update f if necessary
            # if at least one id in blankness is blank them time_idx is updated
            # and we need to update f accordingly
            if self.count_nonzero(blankness) > 0:
                fetch_time_idxs = time_idxs.min(max_lens)
                # select tensor along second dim of x
                # implement something like --> f = x[:, :, fetch_time_idxs, :]
                # for example, if all elements in fetch_time_idxs = n, then
                # this is equivelent to f = x[:, :, n, :]
                f = x[list(range(batch_size)), fetch_time_idxs, :].unsqueeze(1)
        for i in range(batch_size):
            label_copy[i]=label_tensor[i][1:label_col[i]+1].tolist()
        return label_copy

    def _pred_step_batch(self, label, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]], batch_size) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # not really need this line, _blank_id is the last id of dict
        #label = label - label.gt(self._blank_id).int()
        result = self._model.prediction(label, hidden, batch_size)
        return result

    def _joint_step_batch(self, enc: torch.Tensor, pred: torch.Tensor, log_normalize: bool=False) -> torch.Tensor:
        logits = self._model.joint(enc, pred)
        logits = logits[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs
