# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import Cache
from transformers.models.llama.modeling_llama import (
    CausalLMOutputWithPast,
)

from transformers.models.llama.modeling_llama import (
    is_flash_attn_2_available,
    replace_return_docstrings,
    add_start_docstrings_to_model_forward,
    LLAMA_INPUTS_DOCSTRING
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel

_CONFIG_FOR_DOC = "LlamaConfig"

from llava.constants import IGNORE_INDEX


class ForwardKLLoss(torch.nn.Module):
    """
    The Kullback-Leibler divergence loss for valid indexes.
    Implementation of https://github.com/jongwooko/distillm/blob/17c0f98bc263b1861a02d5df578c84aea652ee65/distillm/losses.py

    Args:
        ignore_index (int):  Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).
        """

        teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(student_logits)
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()
        if not normalize:
            return -torch.sum(x * mask.view(-1), dim=0)
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=x.device)
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

class ForwardKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Forward KL with chunked outputs that saves memory by only upcasting one chunk at a time.

    Since the model is trained with bf16, before computing KL divergence, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    result (bsz, num_tokens, vocab_size). If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into. Each chunk has shape
            (batch_size, num_tokens / num_output_chunks, vocab_size).
            Default: 8
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.fkl_loss = ForwardKLLoss(ignore_index)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_tokens).

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).

        Example:
            >>> loss_fn = ForwardKLWithChunkedOutputLoss()
            >>>
            >>> h = torch.tensor([bsz, num_tokens, dim])
            >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> teacher_chunks = [teacher_model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> labels = torch.tensor([bsz, num_tokens])
            >>> loss = loss_fn(output_chunks, teacher_chunks, labels)
        """

        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        mask = (labels != self.ignore_index).int()
        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        total_fkl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_fkl_loss += self.fkl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_fkl_loss / torch.sum(mask.view(-1), dim=0)



class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        multi_modal_index: Optional[List[int]] = None,
        pure_text_index: Optional[List[int]] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            return_dict:
            cache_position:
            position_ids:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        # Placeholder for returns (even if the function does not return anything or returns a specific type)

        ```python
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb;pdb.set_trace()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        if multi_modal_index is None:
            multi_modal_index = []
        if pure_text_index is None:
            pure_text_index = []

        # LLaVA 损失和蒸馏损失计算
        loss_fct = CrossEntropyLoss()
        loss_fkl = ForwardKLWithChunkedOutputLoss(ignore_index=IGNORE_INDEX)

        llava_loss = None
        kd_loss = None

        # import pdb;pdb.set_trace()
        # LLaVA 损失计算
        # if len(multi_modal_index) > 0:
        logits_multi_modal = logits.clone()
        labels_multi_modal = labels.clone()


        # 移位处理
        shift_logits = logits_multi_modal[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
        shift_labels = labels_multi_modal[..., 1:].contiguous().view(-1)

        # 计算 LLaVA 损失
        shift_labels = shift_labels.to(shift_logits.device)  # 确保标签在相同的设备上
        llava_loss = loss_fkl(shift_logits,shift_logits, shift_labels)

        # 蒸馏损失计算
        if len(pure_text_index) > 0:
            # 获取旧模型输出
            # outputs_old = self.model_old(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     inputs_embeds=inputs_embeds,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            # output1 = self.model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_values=past_key_values,
            #     inputs_embeds=inputs_embeds,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            output2 = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # import pdb;pdb.set_trace()

            hidden_states_old = output2[0]
            # hidden_states_old = outputs_old[0]

            # 计算旧模型的 logits
            if self.config.pretraining_tp > 1:
                lm_head_slices_old = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits_old = [F.linear(hidden_states_old, lm_head_slices_old[i]) for i in
                              range(self.config.pretraining_tp)]
                logits_old = torch.cat(logits_old, dim=-1)
            else:
                logits_old = self.lm_head(hidden_states_old)

            logits_pure_text = logits[pure_text_index].clone()
            logits_pure_text_old = logits_old[pure_text_index].clone()
            labels_pure_text = labels[pure_text_index].clone()

            # 移位处理
            shift_logits_new = logits_pure_text[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_logits_old = logits_pure_text_old[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels_text = labels_pure_text[..., 1:].contiguous().view(-1)

            # 计算蒸馏损失
            shift_labels_text = shift_labels_text.to(shift_logits_new.device)  # 确保标签在相同设备上
            kd_loss = loss_fkl(
                student_logits=shift_logits_new,
                teacher_logits=shift_logits_old,
                labels=shift_labels_text
            )

        # import pdb;pdb.set_trace()
        if kd_loss is not None and llava_loss is not None:
            loss = kd_loss * 10.0 + llava_loss
        elif kd_loss is not None:
            loss = kd_loss * 10.0
        elif llava_loss is not None:
            loss = llava_loss
        else:
            loss = None  # 如果两个损失都没有，设置为 None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
