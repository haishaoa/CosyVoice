# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Generator
import json
import onnxruntime
import torch
import numpy as np
import whisper
from typing import Callable
import torchaudio.compliance.kaldi as kaldi
import os
import re
import inflect
from cosyvoice.utils.file_utils import logging, load_wav
from cosyvoice.utils.frontend_utils import (
    contains_chinese,
    replace_blank,
    replace_corner_mark,
    remove_bracket,
    spell_out_number,
    split_paragraph,
    is_only_punctuation,
)


class CosyVoiceFrontEnd:

    def __init__(
        self,
        get_tokenizer: Callable,
        feat_extractor: Callable,
        campplus_model: str,
        speech_tokenizer_model: str,
        spk2info: str = "",
        allowed_special: str = "all",
    ):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model,
            sess_options=option,
            providers=[
                (
                    "CUDAExecutionProvider"
                    if torch.cuda.is_available()
                    else "CPUExecutionProvider"
                )
            ],
        )
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)
        else:
            self.spk2info = {}
        self.allowed_special = allowed_special
        self.inflect_parser = inflect.engine()
        # NOTE compatible when no text frontend tool is avaliable
        try:
            import ttsfrd

            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert (
                self.frd.initialize(
                    "{}/../../pretrained_models/CosyVoice-ttsfrd/resource".format(
                        ROOT_DIR
                    )
                )
                is True
            ), "failed to initialize ttsfrd resource"
            self.frd.set_lang_type("pinyinvg")
            self.text_frontend = "ttsfrd"
            logging.info("use ttsfrd frontend")
        except:
            try:
                from wetext import Normalizer as ZhNormalizer
                from wetext import Normalizer as EnNormalizer

                self.zh_tn_model = ZhNormalizer(remove_erhua=False)
                self.en_tn_model = EnNormalizer()
                self.text_frontend = "wetext"
                logging.info("use wetext frontend")
            except:
                self.text_frontend = ""
                logging.info("no frontend is avaliable")

    def _extract_text_token(self, text):
        """
        提取文本中的token
        """

        # 检查输入的text是否是生成器（generator）类型
        if isinstance(text, Generator):
            # 记录日志
            logging.info(
                "get tts_text generator, will return _extract_text_token_generator!"
            )
            # NOTE add a dummy text_token_len for compatibility
            # 返回一个元组：第一个元素是调用专门处理生成器的函数，第二个元素是占位用的长度为0的tensor（为了保持返回格式一致）
            return self._extract_text_token_generator(text), torch.tensor(
                [0], dtype=torch.int32
            ).to(self.device)
        else:
            # 使用tokenizer将文本编码为token序列，允许某些特殊字符
            text_token = self.tokenizer.encode(
                text, allowed_special=self.allowed_special
            )
            # 将token列表转换为PyTorch张量，并移动到指定的设备（self.device）
            text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
            # 计算token序列的长度，并转换为张量
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(
                self.device
            )
            # 返回文本token张量和其长度张量
            return text_token, text_token_len

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i : i + 1]

    def _extract_speech_token(self, prompt_wav):
        """
        提取语音token

        :param self: 说明
        :param prompt_wav: 音频路径
        """

        # 加载音频文件，从采样为16kHz，返回音频数据的张量
        speech = load_wav(prompt_wav, 16000)
        # 检查音频长度：音频采样点数/采样率=秒数，确保不超过3 0秒
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"
        # 使用whisper库提取音频的对数梅尔频谱特征，使用128个梅尔滤波器组
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = (
            # 使用onnxruntime进行推理，返回语音token
            self.speech_tokenizer_session.run(
                None,
                {
                    self.speech_tokenizer_session.get_inputs()[0]
                    # 第一个输入：梅尔频谱特征（转换为numpy数组）
                    .name: feat.detach().cpu().numpy(),
                    self.speech_tokenizer_session.get_inputs()[1].name: np.array(
                        # 第二个输入：特征序列长度（时间帧数）
                        [feat.shape[2]],
                        dtype=np.int32,
                    ),
                },
            )[0]
            # 将多维数据展平为一维
            .flatten()
            # 转换为python列表
            .tolist()
        )
        # 张量转换和设备移动
        speech_token = torch.tensor([speech_token], dtype=torch.int32).to(self.device)
        # 创建表示token序列长度的张量
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(
            self.device
        )
        # 返回token序列（整数ID和token序列长度）
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        # 加载音频文件
        speech = load_wav(prompt_wav, 16000)
        # 提取声学特征
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        # 特征归一化
        feat = feat - feat.mean(dim=0, keepdim=True)
        # 提取说话人嵌入
        embedding = (
            # 运行ONNX模型推理
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        # 转换为PyTorch张量，并转移到指定设备
        embedding = torch.tensor([embedding]).to(self.device)
        # 返回说话人嵌入
        return embedding

    def _extract_speech_feat(self, prompt_wav):
        """
        提取音频的声学特征

        :param self: 当前类的实例对象
        :param prompt_wav: 音频文件路径
        """

        # 加载并将音频重采样到24000Hz，返回一个原始波形数据（一维数组）
        speech = load_wav(prompt_wav, 24000)
        speech_feat = (
            # feat_extractor:预定义的梅尔频谱（Mel Spectrogram）提取器
            self.feat_extractor(speech)
            # 移除第0维
            .squeeze(dim=0)
            .transpose(0, 1)
            .to(self.device)
        )
        # 添加第0维
        speech_feat = speech_feat.unsqueeze(dim=0)
        # 获取特征序列的长度
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(
            self.device
        )
        # 1、返回声学特征张量 2、特征序列长度的张量
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info("get tts_text generator, will skip text_normalize!")
            return [text]
        # NOTE skip text_frontend when ssml symbol in text
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split is True else text
        text = text.strip()
        if self.text_frontend == "ttsfrd":
            texts = [
                i["text"]
                for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]
            ]
            text = "".join(texts)
        else:
            if contains_chinese(text):
                if self.text_frontend == "wetext":
                    text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
                texts = list(
                    split_paragraph(
                        text,
                        partial(
                            self.tokenizer.encode, allowed_special=self.allowed_special
                        ),
                        "zh",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
            else:
                if self.text_frontend == "wetext":
                    text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(
                    split_paragraph(
                        text,
                        partial(
                            self.tokenizer.encode, allowed_special=self.allowed_special
                        ),
                        "en",
                        token_max_n=80,
                        token_min_n=60,
                        merge_len=20,
                        comma_split=False,
                    )
                )
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token,
            "text_len": tts_text_token_len,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        """
        Args:
            tts_text:待合成文本
            prompt_text:提示文本
            prompt_wav:提示音频
            resample_rate:重采样率
            zero_shot_spk_id:零样本说话人ID
        """

        # 提取待合成文本的token序列及其长度
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        # 判断是否使用预先注册的说话人
        if zero_shot_spk_id == "":
            # 提取提示文本的token序列及其长度
            prompt_text_token, prompt_text_token_len = self._extract_text_token(
                prompt_text
            )
            # speech_feat：管说的音色   speech_token：管说的内容
            # 提取提示音频的：1、声学特征张量 2、特征序列长度的张量
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
            # 提取提示音频的语音token和长度
            speech_token, speech_token_len = self._extract_speech_token(prompt_wav)
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                # 确保声学特征长度是语音token长度的2倍
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = (
                    speech_feat[:, : 2 * token_len],
                    2 * token_len,
                )
                speech_token, speech_token_len[:] = (
                    speech_token[:, :token_len],
                    token_len,
                )
            # 声学指纹提取说话人嵌入向量：控制说话人的身份
            embedding = self._extract_spk_embedding(prompt_wav)
            model_input = {
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat,
                "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        else:
            model_input = self.spk2info[zero_shot_spk_id]
        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        return model_input

    def frontend_cross_lingual(
        self, tts_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        model_input = self.frontend_zero_shot(
            tts_text, "", prompt_wav, resample_rate, zero_shot_spk_id
        )
        # in cross lingual mode, we remove prompt in llm
        del model_input["prompt_text"]
        del model_input["prompt_text_len"]
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input["llm_embedding"]
        instruct_text_token, instruct_text_token_len = self._extract_text_token(
            instruct_text
        )
        model_input["prompt_text"] = instruct_text_token
        model_input["prompt_text_len"] = instruct_text_token_len
        return model_input

    def frontend_instruct2(
        self, tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        model_input = self.frontend_zero_shot(
            tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
        )
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_wav, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(
            prompt_wav
        )
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(
            prompt_wav
        )
        embedding = self._extract_spk_embedding(prompt_wav)
        source_speech_token, source_speech_token_len = self._extract_speech_token(
            source_speech_16k
        )
        model_input = {
            "source_speech_token": source_speech_token,
            "source_speech_token_len": source_speech_token_len,
            "flow_prompt_speech_token": prompt_speech_token,
            "flow_prompt_speech_token_len": prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat,
            "prompt_speech_feat_len": prompt_speech_feat_len,
            "flow_embedding": embedding,
        }
        return model_input
