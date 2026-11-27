"""Live classroom interpreter — 麦克风 -> Whisper -> 翻译 -> 悬浮字幕窗。

运行：
    D:\\asr-lab\\venv\\Scripts\\python.exe D:\\asr-lab\\overlay.py
    D:\\asr-lab\\venv\\Scripts\\python.exe D:\\asr-lab\\overlay.py --model small
    D:\\asr-lab\\venv\\Scripts\\python.exe D:\\asr-lab\\overlay.py --file D:\\asr-lab\\work\\test_en.wav

窗口操作：
    拖边缘 / 拖四角   自由缩放（八方向）
    拖窗口内部       移动
    Ctrl+滚轮        字体大小
    顶部按钮         ‖ 暂停   ◌ 点击穿透   ✕ 关闭
    窗口大小和位置会记住，下次打开还原
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import queue
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QIcon, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSystemTrayIcon,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

DARK_QSS = """
QDialog { background: #171A20; }
QLabel { color: #D5DCE8; background: transparent; }
QComboBox, QLineEdit {
    background: #22262E; color: #E8EDF5;
    border: 1px solid #2E3440; border-radius: 8px; padding: 6px 9px;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #22262E; color: #E8EDF5;
    selection-background-color: #2F3947; border: 1px solid #2E3440;
}
QCheckBox { color: #C7D0DD; }
QPushButton {
    background: #232833; color: #DCE3EE;
    border: 1px solid #303743; border-radius: 8px; padding: 6px 18px;
}
QPushButton:hover { background: #2A313D; }
QPushButton:default { background: #2563EB; border-color: #2563EB; color: white; }
"""

if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))

# 让 Windows 把这个进程当成独立应用，任务栏才认窗口图标而不是 pythonw.exe 的
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "CaptiNote.Desktop.1"
        )
    except Exception:
        pass
MODEL_DIR = os.path.join(ROOT, "models")
UI_STATE = os.path.join(ROOT, "ui.json")
SR = 16000
BLOCK = 1600  # 100 ms

EDGE = 9           # 边缘感应宽度
MIN_W, MIN_H = 340, 130

# 提示词会被 Whisper 漏进输出（听不清时拿它填空），所以这里不塞术语表。
# 术语只能靠翻译层，或识别后纠正。
DEFAULT_PROMPT = "English lecture."

# 低于这个电平一律当静音，防止自适应底噪一路探底、把噪声当说话
ABS_FLOOR_DB = -52.0


# --------------------------------------------------------------------------- #
# 翻译
# --------------------------------------------------------------------------- #
class Translator:
    name = "off"

    def translate(self, text: str, context: str = "") -> str:
        return ""


def _load_config() -> dict:
    """读 D:\\asr-lab\\config.json，改完重启生效。"""
    try:
        with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_glossary() -> dict[str, str]:
    """翻完之后按术语表纠错。文件是 D:\\asr-lab\\glossary.json。"""
    try:
        with open(os.path.join(ROOT, "glossary.json"), "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items() if k and v}
    except Exception:
        return {}


class MyMemory(Translator):
    """免费、无 key，有每日额度，只适合测试。"""

    name = "mymemory"

    def translate(self, text: str, context: str = "") -> str:
        if not text.strip():
            return ""
        q = urllib.parse.quote(text[:480])
        url = f"https://api.mymemory.translated.net/get?q={q}&langpair=en|zh-CN"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.load(resp)
            out = (data.get("responseData") or {}).get("translatedText") or ""
            return out.strip()
        except Exception as exc:
            return f"[翻译失败] {exc.__class__.__name__}"


class LocalMarian(Translator):
    """本地英中翻译：ONNX 版 Marian，不需要 torch，约 160MB。"""

    name = "local"

    def __init__(self, model_dir: str | None = None):
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        import onnxruntime as ort
        from transformers import MarianTokenizer

        d = model_dir or os.path.join(ROOT, "models", "opus-onnx")
        self.tok = MarianTokenizer.from_pretrained(d)
        cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        self.pad = int(cfg.get("pad_token_id", 65000))
        self.eos = int(cfg.get("eos_token_id", 0))

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.log_severity_level = 4
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(
            os.path.join(d, "onnx", "encoder_model_quantized.onnx"), so, providers=prov
        )
        self.dec = ort.InferenceSession(
            os.path.join(d, "onnx", "decoder_model_quantized.onnx"), so, providers=prov
        )
        self.dwp = ort.InferenceSession(
            os.path.join(d, "onnx", "decoder_with_past_model_quantized.onnx"), so, providers=prov
        )
        self._dec_out = [o.name for o in self.dec.get_outputs()][1:]
        self._dwp_out = [o.name for o in self.dwp.get_outputs()][1:]
        self._dwp_in = {i.name for i in self.dwp.get_inputs()}
        self.gloss = _load_glossary()

    def _run(self, text: str) -> str:
        import numpy as np

        ids = self.tok.encode(text)[:480]
        input_ids = np.array([ids], dtype=np.int64)
        mask = np.ones_like(input_ids, dtype=np.int64)
        hidden = self.enc.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]

        out = self.dec.run(
            None,
            {
                "input_ids": np.array([[self.pad]], dtype=np.int64),
                "encoder_attention_mask": mask,
                "encoder_hidden_states": hidden,
            },
        )
        logits = out[0]
        past = dict(zip(self._dec_out, out[1:]))

        got: list[int] = []
        nxt = int(np.argmax(logits[0, -1]))
        for _ in range(160):
            if nxt == self.eos:
                break
            got.append(nxt)
            feed = {
                "input_ids": np.array([[nxt]], dtype=np.int64),
                "encoder_attention_mask": mask,
            }
            for k, v in past.items():
                nm = k.replace("present", "past_key_values")
                if nm in self._dwp_in:
                    feed[nm] = v
            out = self.dwp.run(None, feed)
            logits = out[0]
            for k, v in zip(self._dwp_out, out[1:]):
                past[k] = v
            nxt = int(np.argmax(logits[0, -1]))
        return self.tok.decode(got, skip_special_tokens=True).strip()

    def translate(self, text: str, context: str = "") -> str:
        if not text.strip():
            return ""
        out = self._run(text)
        for bad, good in self.gloss.items():
            out = out.replace(bad, good)
        return out


class DeepSeek(Translator):
    """留好的接口：填 key 就自动切过来。"""

    name = "deepseek"

    def __init__(self, key: str, model: str = "deepseek-chat", glossary: str = "",
                 target: str = "zh"):
        self.key = key
        self.model = model
        self.glossary = (glossary or "").strip()
        self.target = target

    def _lang_word(self) -> str:
        return {"en": "英文", "zh": "中文"}.get(self.target, "中文")

    def _gloss_block(self) -> str:
        if not self.glossary:
            return ""
        return (
            "\n本课已知背景与词表（识别可能听错，纠错时优先往这些词上靠）：\n"
            + self.glossary[:1200]
        )

    def translate_rich(self, text: str, context: str = "") -> tuple[str, str]:
        """一次调用同时返回「纠正后的英文」和「中文翻译」。"""
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"你是课堂同传。把老师说的英文翻成简洁、口语化的{self._lang_word()}。\n"
                            '只输出 JSON：{"fix": "纠正后的英文", "zh": "译文"}\n'
                            "规则：\n"
                            "1. 只改明显听错的词（同音词、术语、数字、周次），不要改写句式、不要补内容\n"
                            "2. 拿不准就原样保留，宁可不改\n"
                            "2a. 参考上文语境理解本句；若上文明显译错，本句按语境译对，别跟着错\n"
                            "2b. fix 字段要补全标点和大小写：该断句的地方加逗号句号问号，"
                            "首字母大写；原文没有标点时也要补上\n"
                            f"3. {self._lang_word()}简洁口语化，术语首次出现写「译名（English）」\n"
                            "4. 这是课堂口语：AI 指人工智能，class 指课不指班级\n"
                            "5. 省略句、代词结合前文补全"
                            + self._gloss_block()
                        ),
                    },
                    {
                        "role": "user",
                        "content": (f"前文：{context}\n\n本句：{text}" if context else text),
                    },
                ],
                "temperature": 0,
                "stream": False,
                "response_format": {"type": "json_object"},
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        obj = json.loads(data["choices"][0]["message"]["content"])
        fixed = str(obj.get("fix") or text).strip()
        zh = str(obj.get("zh") or "").strip()
        if not zh:
            raise ValueError("empty zh")
        return fixed, zh

    def translate(self, text: str, context: str = "") -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            f"你是课堂同传。把老师说的英文翻成简洁、口语化的{self._lang_word()}。"
                            f"术语首次出现写「译名（English）」；拿不准的词保留英文；"
                            "不要总结、不要省略、不要补充。"
                            "这是课堂口语：AI 指人工智能，class 指课不是班级。"
                            "遇到省略句、代词、半句话，结合前文补全，不要逐词硬译。"
                            + self._gloss_block()
                        ),
                    },
                    {
                        "role": "user",
                        "content": (f"前文：{context}\n\n本句：{text}" if context else text),
                    },
                ],
                "temperature": 0,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            return f"[翻译失败] {exc.__class__.__name__}"


# --------------------------------------------------------------------------- #
# 断句
# --------------------------------------------------------------------------- #
class VadSegmenter:
    """Silero VAD 断句：按语音概率切，比能量阈值稳得多。

    每 200ms 对最近 512ms 音频算一次语音概率，用滞回判断起止。
    参数可在 config.json 里调：seg_silence_ms / seg_min_s / seg_max_s
    """

    FRAME = 512
    WINDOW = 8192

    def __init__(self, silence_ms=None, max_ms=None, pre_roll_ms=300):
        _cfg = _load_config()
        if silence_ms is None:
            silence_ms = int(_cfg.get("seg_silence_ms", 700))
        if max_ms is None:
            max_ms = int(float(_cfg.get("seg_max_s", 10)) * 1000)
        self.min_n = int(float(_cfg.get("seg_min_s", 5)) * SR)
        from faster_whisper.vad import get_vad_model

        self.model = get_vad_model()
        self.acc = np.zeros(0, dtype=np.float32)
        self.pend = np.zeros(0, dtype=np.float32)
        self.pre = np.zeros(0, dtype=np.float32)
        self.pre_n = SR * pre_roll_ms // 1000
        self.silence_n = silence_ms
        self.max_n = SR * max_ms // 1000
        self.speaking = False
        self.seg: list[np.ndarray] = []
        self.seg_len = 0
        self.silence = 0

    def _prob(self) -> float:
        w = self.acc[-self.WINDOW:]
        n = (len(w) // self.FRAME) * self.FRAME
        if n < self.FRAME * 6:
            return 0.0
        p = self.model(np.ascontiguousarray(w[-n:], dtype=np.float32))
        return float(np.mean(p[-4:]))

    def _emit(self):
        if not self.seg:
            self.speaking = False
            return None
        audio = np.concatenate(self.seg).astype(np.float32)
        self.seg = []
        self.seg_len = 0
        self.silence = 0
        self.speaking = False
        return audio

    def feed(self, chunk: np.ndarray):
        chunk = chunk.astype(np.float32, copy=False)
        self.acc = np.concatenate((self.acc, chunk))[-SR * 2:]
        self.pend = np.concatenate((self.pend, chunk))
        if len(self.pend) < SR // 5:  # 200 ms
            return None
        self.pend = np.zeros(0, dtype=np.float32)

        prob = self._prob()

        if not self.speaking:
            self.pre = np.concatenate((self.pre, chunk))[-self.pre_n:]
            if prob > 0.45:
                self.speaking = True
                self.seg = [self.pre.copy()]
                self.seg_len = len(self.pre)
                self.silence = 0
            return None

        self.seg.append(chunk)
        self.seg_len += len(chunk)
        self.silence = 0 if prob > 0.30 else self.silence + SR // 5

        if self.silence >= self.silence_n or self.seg_len >= self.max_n:
            # 短段快出字；上下文由滑动窗口补足，见 submit(ctx=...)
            if self.seg_len >= self.min_n:
                return self._emit()
            self.silence = 0
        return None


class Segmenter:
    """能量 VAD：自适应底噪 + 静音切分 + 超长强制切段。"""

    def __init__(self, silence_ms=600, max_ms=6000, pre_roll=3, margin_db=7.0):
        self.silence_frames = max(1, silence_ms // 100)
        self.max_frames = max(2, max_ms // 100)
        self.pre_roll = pre_roll
        self.margin_db = margin_db
        self.noise_db = -55.0
        self.speaking = False
        self.buf: list[np.ndarray] = []
        self.pre: list[np.ndarray] = []
        self.quiet = 0
        self.hits = 0

    @staticmethod
    def _db(chunk: np.ndarray) -> float:
        return float(20.0 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-9))

    def feed(self, chunk: np.ndarray):
        db = self._db(chunk)
        thresh = max(self.noise_db + self.margin_db, ABS_FLOOR_DB)

        if not self.speaking:
            # 安静时持续更新底噪，但不许探到无声以下
            if db < thresh:
                self.noise_db = max(-75.0, 0.97 * self.noise_db + 0.03 * db)
            self.pre.append(chunk)
            if len(self.pre) > self.pre_roll:
                self.pre.pop(0)
            if db > thresh:
                self.hits += 1
                if self.hits >= 2:
                    self.speaking = True
                    self.buf = list(self.pre)
                    self.quiet = 0
            else:
                self.hits = 0
            return None

        self.buf.append(chunk)
        self.quiet = self.quiet + 1 if db < thresh else 0
        if self.quiet >= self.silence_frames or len(self.buf) >= self.max_frames:
            audio = np.concatenate(self.buf).astype(np.float32)
            self.speaking = False
            self.buf = []
            self.pre = []
            self.quiet = 0
            self.hits = 0
            return audio
        return None


# --------------------------------------------------------------------------- #
# 识别 / 翻译线程
# --------------------------------------------------------------------------- #
INT8_DIR = os.path.join(ROOT, "models", "turbo-int8")


def _resolve_model(name: str) -> str:
    """有 int8 量化版就用它，体积减半、加载更快，输出一致。"""
    if name == "large-v3-turbo" and os.path.isdir(INT8_DIR):
        return INT8_DIR
    return name


_GLOSS_STOP = {
    "the", "and", "for", "with", "ai", "lab", "week", "two", "six",
    "one", "yes", "ok", "okay", "you", "are", "this", "that", "from",
    "not", "our", "his", "her", "they", "can",
}

_FILLER_CHARS = "呃啊嗯额哦唔唉哎噢喔"
_FILLER_WORDS = {"uh", "um", "ah", "er", "mm", "hmm", "oh", "eh", "erm", "hm"}


def _is_filler_only(s: str) -> bool:
    """整句只有语气词（uh/呃…）时返回 True，这种不该单独上屏。"""
    t = re.sub(f"[{_FILLER_CHARS}]", " ", (s or "").lower())
    t = re.sub(r"[^\w\s']", " ", t)
    words = [w for w in t.split() if w]
    return not words or all(w in _FILLER_WORDS for w in words)


_FILLER_MAP = {
    "呃": "uh", "额": "uh", "啊": "ah", "嗯": "um",
    "哦": "oh", "噢": "oh", "喔": "oh", "唔": "mm",
    "唉": "ah", "哎": "ah",
}


LOG_DIR = os.path.join(ROOT, "logs")


def _log(msg: str) -> None:
    """写一行运行日志，出问题时靠它定位。"""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, "latest.log")
        if os.path.exists(path) and os.path.getsize(path) > 2 * 1024 * 1024:
            os.replace(path, os.path.join(LOG_DIR, "prev.log"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")
    except Exception:
        pass



def _sentence_case(s: str) -> str:
    # 双语模型把 uh/ah/um 写成了汉字，这里映射回英文而不是删掉，
    # 老师真说的中文词保留原样（那是信息，不该丢）
    for k, v in _FILLER_MAP.items():
        if k in s:
            s = s.replace(k, f" {v} ")
    s = " ".join(s.split()).lower()
    return s[:1].upper() + s[1:] if s else s


class DraftStreamer:
    """流式草稿：边说边出，攒够一定长度或停顿够久才输出，避免碎成一片。"""

    def __init__(self, model_dir: str | None = None):
        import sherpa_onnx

        d = model_dir or os.path.join(ROOT, "models", "stream-zip-bi")
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(d, "tokens.txt"),
            encoder=os.path.join(d, "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=os.path.join(d, "decoder-epoch-99-avg-1.int8.onnx"),
            joiner=os.path.join(d, "joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=2,
            sample_rate=SR,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=1.6,
            rule2_min_trailing_silence=1.0,
            decoding_method="greedy_search",
            provider="cpu",
        )
        self.stream = self.rec.create_stream()
        self.on_draft = None       # callable(str en)
        self._last_text = ""
        self._last_change = time.time()
        self._last_tx = 0.0

    def reset(self):
        try:
            self.rec.reset(self.stream)
        except Exception:
            pass
        self._last_text = ""
        self._last_change = time.time()

    def feed(self, chunk: np.ndarray):
        self.stream.accept_waveform(SR, chunk)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)

        cur = self.rec.get_result(self.stream).strip()
        now = time.time()

        # 句末：交给定稿，然后清空重新攒下一句
        if self.rec.is_endpoint(self.stream) and cur:
            text = _sentence_case(cur)
            if text and not _is_filler_only(text) and self.on_draft:
                self.on_draft(text)
            self.reset()
            return

        # 说话中：约每 250ms 把当前 partial 整块刷新一次（逐词跟手）
        if cur and cur != self._last_text and (now - self._last_tx) > 0.25:
            self._last_text = cur
            self._last_tx = now
            text = _sentence_case(cur)
            if text and not _is_filler_only(text) and self.on_draft:
                self.on_draft(text)


class Recognizer:
    def __init__(self, model_name: str, threads: int, prompt: str,
                 language: str = "en", engine: str = "whisper"):
        self.model_name = model_name
        self.threads = threads
        self.prompt = prompt
        self.language = language or "en"
        self.engine = engine
        self.para = None
        self.model: WhisperModel | None = None
        self.jobs: queue.Queue = queue.Queue()
        self.on_text = None
        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._work, daemon=True).start()

    def _load(self):
        if self.engine == "parakeet":
            try:
                import sherpa_onnx

                d = os.path.join(MODEL_DIR, "parakeet-v2-int8")
                self.para = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=os.path.join(d, "encoder.int8.onnx"),
                    decoder=os.path.join(d, "decoder.int8.onnx"),
                    joiner=os.path.join(d, "joiner.int8.onnx"),
                    tokens=os.path.join(d, "tokens.txt"),
                    num_threads=self.threads,
                    sample_rate=SR,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    model_type="nemo_transducer",
                    provider="cpu",
                )
                _log("识别引擎 Parakeet 已加载")
                return
            except Exception as exc:
                _log(f"Parakeet 加载失败 {exc.__class__.__name__}: {exc}，退回 Whisper")
                self.engine = "whisper"
        try:
            self.model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
                download_root=MODEL_DIR,
                cpu_threads=self.threads,
            )
            self.load_error = None
        except Exception as exc:
            import traceback

            self.load_error = f"{exc.__class__.__name__}: {exc}"
            traceback.print_exc()
            print(f"[ASR] 模型加载失败: {self.load_error}", flush=True)

    def submit(self, audio: np.ndarray, ctx: np.ndarray | None = None,
               skip: float = 0.0):
        """skip: 窗口内早于这个秒数的词已经输出过，不再重复。"""
        if self.jobs.qsize() < 4:
            self.jobs.put((audio, ctx, skip))

    def _work(self):
        while True:
            audio, ctx, skip = self.jobs.get()
            while self.model is None and self.para is None:
                time.sleep(0.2)
            ctx_n = 0 if ctx is None else len(ctx)
            feed = audio if not ctx_n else np.concatenate([ctx, audio])
            if self.engine == "parakeet" and self.para is not None:
                try:
                    st = self.para.create_stream()
                    st.accept_waveform(SR, audio)  # Parakeet 不用上下文，避免重复
                    self.para.decode_stream(st)
                    text = st.result.text.strip()
                    if text:
                        self.on_text(text)
                        _log(f"ASR(par) ok {len(feed)/SR:.1f}s → {len(text)} 字")
                    continue
                except Exception as exc:
                    _log(f"Parakeet 识别失败 {exc.__class__.__name__}")
                    continue
            try:
                segs, _ = self.model.transcribe(
                    feed,
                    language=self.language,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt=self.prompt or None,
                    word_timestamps=bool(ctx_n),
                )
                # 低置信度的段直接丢：Whisper 在没人说话时会编台词
                kept = [
                    s for s in segs
                    if s.no_speech_prob < 0.8 and s.avg_logprob > -1.5
                ]
                if ctx_n:
                    # 按绝对时间切：早于 skip 的词已经输出过
                    cut = max(0.0, skip)
                    words = [w.word for s in kept for w in (s.words or [])
                             if w.start >= cut]
                    text = " ".join(" ".join(words).split())
                    if not text:  # 没有词级时间戳就退回整段
                        text = " ".join(s.text.strip() for s in kept).strip()
                else:
                    text = " ".join(s.text.strip() for s in kept).strip()
                text = _dedupe(text)
                if text.strip().lower() in _HALLUCINATIONS:
                    text = ""
            except Exception as exc:
                import traceback

                traceback.print_exc()
                print(f"[ASR] 识别失败: {exc.__class__.__name__}: {exc}", flush=True)
                _log(f"ASR 失败 {exc.__class__.__name__}: {exc}")
                text = ""
            if text and self.on_text:
                self.on_text(text)
                _log(f"ASR ok {len(audio)/SR:.1f}s → {len(text)} 字")


class TranslatorWorker:
    def __init__(self, factory):
        self.factory = factory
        self.tr = None
        self.error: str | None = None
        self.ready = threading.Event()
        self.jobs: queue.Queue = queue.Queue()
        self.history: list[tuple[str, str]] = []
        self.on_text = None
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, text: str):
        if self.jobs.qsize() < 3:
            self.jobs.put(text)

    def reload(self, factory):
        self.factory = factory
        self.tr = None
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.tr = self.factory()
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}"

    def _work(self):
        self._load()
        self.ready.set()
        while True:
            text = self.jobs.get()
            if self.tr is None:
                if self.on_text:
                    self.on_text(text, f"[翻译不可用] {self.error or ''}")
                continue
            ctx = " | ".join(f"{e} → {z}" for e, z in self.history[-4:])
            fixed, out = text, ""
            rich = getattr(self.tr, "translate_rich", None)
            try:
                if rich is not None:
                    fixed, out = rich(text, ctx)
                else:
                    out = self.tr.translate(text, ctx)
            except Exception:
                out = ""
                fixed = text
            # 主引擎挂了或者返回失败串，自动换免费引擎兜底
            if (not out) or out.startswith("[翻译失败"):
                try:
                    if getattr(self, "_fb", None) is None:
                        self._fb = MyMemory()
                    alt = self._fb.translate(text, ctx)
                    if alt:
                        out = alt
                        _log("翻译兜底 MyMemory 生效")
                except Exception as exc:
                    _log(f"翻译兜底也失败 {exc.__class__.__name__}")
            if out and self.on_text:
                self.history.append((fixed, out))
                if len(self.history) > 10:
                    self.history.pop(0)
                self.on_text(fixed, out)


class FadeOut(QWidget):
    """顶部渐隐，避免被切一半的句子硬生生露出来。"""

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._c = QColor(color)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def set_color(self, color: QColor):
        self._c = QColor(color)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        g = QLinearGradient(0, 0, 0, self.height())
        a = QColor(self._c)
        b = QColor(self._c)
        b.setAlpha(0)
        g.setColorAt(0.0, a)
        g.setColorAt(0.55, QColor(b.red(), b.green(), b.blue(), int(a.alpha() * 0.62)))
        g.setColorAt(1.0, b)
        p.fillRect(self.rect(), g)
        p.end()


class LevelMeter(QWidget):
    """一排小竖条，显示麦克风进来的音量。"""

    def __init__(self, n: int = 20, parent=None):
        super().__init__(parent)
        self.n = n
        self.vals = [0.0] * n
        self.setFixedSize(104, 14)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._color = QColor("#4ade80")

    def push(self, v: float):
        self.vals = self.vals[1:] + [max(0.0, min(1.0, float(v)))]
        self.update()

    def set_color(self, c: str):
        self._color = QColor(c)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        gap = 2.0
        bw = max(1.6, (w - gap * (self.n - 1)) / self.n)

        track = QColor(self._color)
        track.setAlphaF(0.16)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, h - 2.0, w, 2.0), 1, 1)

        for i, v in enumerate(self.vals):
            bh = max(3.0, h * (0.18 + 0.82 * v))
            col = QColor(self._color)
            col.setAlphaF(0.30 + 0.70 * v)
            p.setBrush(col)
            p.drawRoundedRect(QRectF(i * (bw + gap), h - bh, bw, bh), bw / 2, bw / 2)
        p.end()


# --------------------------------------------------------------------------- #
# 录音 / 记录落盘
# --------------------------------------------------------------------------- #
class Session:
    """一边跑一边把原始录音写成 wav，把双语字幕追加进 md。"""

    def __init__(self, sr: int = SR):
        self._t0 = time.time()
        stamp = datetime.datetime.now().strftime("%m%d_%H%M")
        base = os.path.join(ROOT, "recordings")
        self.dir = os.path.join(base, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.wav = os.path.join(self.dir, "audio.wav")
        self.md = os.path.join(self.dir, "transcript.md")
        self._w = wave.open(self.wav, "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(sr)
        with open(self.md, "w", encoding="utf-8") as f:
            f.write(f"# 课堂记录 {stamp}\n\n")

    def write_audio(self, chunk: np.ndarray) -> None:
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            self._w.writeframes(pcm.tobytes())
        except Exception:
            pass

    def write_line(self, orig: str, trans: str) -> None:
        try:
            import time as _t

            el = int(_t.time() - self._t0)
            stamp = f"{el // 60:02d}:{el % 60:02d}"
            with open(self.md, "a", encoding="utf-8") as f:
                f.write(f"[{stamp}] **{orig}**\n\n{trans}\n\n")
        except Exception:
            pass

    def read_text(self) -> str:
        try:
            with open(self.md, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def close(self) -> None:
        try:
            self._w.close()
        except Exception:
            pass

    def compress(self) -> None:
        """wav 转 opus（体积约 1/5），成功后删掉 wav。"""
        try:
            import av

            dst = self.wav[:-4] + ".opus"
            inp = av.open(self.wav)
            ist = inp.streams.audio[0]
            out = av.open(dst, "w")
            ost = out.add_stream("libopus", rate=16000)
            ost.layout = "mono"
            res = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=16000
            )
            for frame in inp.decode(ist):
                for f in res.resample(frame):
                    for pkt in ost.encode(f):
                        out.mux(pkt)
            for pkt in ost.encode(None):
                out.mux(pkt)
            out.close()
            inp.close()
            os.remove(self.wav)
        except Exception:
            pass

    def discard(self) -> None:
        """丢弃这次录音：关文件、删掉 wav 和 md。"""
        self.close()
        for p in (self.wav, self.md):
            try:
                os.remove(p)
            except Exception:
                pass


def llm_translate_batch(lines: list[str], key: str, glossary: str = "") -> list[str]:
    """一次翻一批句子，返回等长中文列表；失败抛异常。"""
    import urllib.request

    payload = json.dumps(lines, ensure_ascii=False)
    sys_p = (
        "你是课堂同传译者。输入是英文句子数组，按顺序译成简体中文，"
        "输出同长度的 JSON 数组，只输出数组。上下文是同一节课的连续讲授，"
        "注意术语一致、指代明确。\n"
    )
    if glossary:
        sys_p += "本课术语（优先采用）：\n" + glossary[:1200] + "\n"
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": payload},
        ],
        "temperature": 0.2,
        "max_tokens": 2600,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    txt = data["choices"][0]["message"]["content"].strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[txt.find("["):]
    arr = json.loads(txt)
    if not isinstance(arr, list) or len(arr) != len(lines):
        raise ValueError("数量对不上")
    return [str(x) for x in arr]


def llm_summary(text: str, key: str, target: str = "zh") -> dict:
    lang = "中文" if target == "zh" else "English"
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "你是课堂笔记助手。只依据给定记录整理，不要编造。只输出 JSON。",
                },
                {
                    "role": "user",
                    "content": (
                        f"下面是一节课的双语记录，每段前有 [mm:ss] 时间戳。用{lang}输出 JSON：\n"
                        '{"title": "不超过16字", "tldr": "一句话概括", '
                        '"timeline": [{"start": "mm:ss", "end": "mm:ss", "title": "小标题", '
                        '"points": ["要点"]}], '
                        '"sections": [{"heading": "分栏标题", "bullets": ["要点"]}], '
                        '"todos": [{"who": "谁", "task": "做什么", "when": "时限"}], '
                        '"terms": [{"en": "English term", "zh": "中文"}]}\n'
                        "时间轴按内容分 3-5 段，用记录里的时间戳；sections 2-4 栏；"
                        "todos 没有就空数组；terms 抽 10-20 个本节课的关键术语/生词。\n\n"
                        + text[:14000]
                    ),
                },
            ],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return json.loads(data["choices"][0]["message"]["content"])


def llm_qa(key: str, transcript: str, summary: str, question: str) -> str:
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "你是助教。只依据给定的课堂记录回答，记录里没有就直说没提到，不要编。回答简洁。",
                },
                {
                    "role": "user",
                    "content": (
                        f"纪要：\n{summary[:4000]}\n\n双语记录：\n{transcript[:9000]}\n\n"
                        f"问题：{question}"
                    ),
                },
            ],
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", s)
    return s.replace("`", "")


def summary_md(obj: dict) -> str:
    out = []
    tldr = obj.get("tldr")
    if tldr:
        out += [f"> {tldr}", ""]
    tl = obj.get("timeline") or []
    if tl:
        out += ["### 时间轴", ""]
        for it in tl:
            st = it.get("start", "")
            head = f"[**{st} – {it.get('end', '')}**](play:{_mmss(st)})　{it.get('title', '')}"
            out.append(head)
            for p in it.get("points") or []:
                out.append(f"- {p}")
            out.append("")
    for s in obj.get("sections") or []:
        out += [f"### {s.get('heading', '')}", ""]
        for b in s.get("bullets") or []:
            out.append(f"- {b}")
        out.append("")
    todos = obj.get("todos") or []
    if todos:
        out += ["### 待办", ""]
        for t in todos:
            who = t.get("who", "")
            task = t.get("task", "")
            when = t.get("when", "")
            line = f"- **{who}**：{task}" if who else f"- {task}"
            if when:
                line += f"（{when}）"
            out.append(line)
        out.append("")
    terms = obj.get("terms") or []
    if terms:
        out += ["### 本节课术语", ""]
        for t in terms:
            en = (t.get("en") or "").strip()
            zh = (t.get("zh") or "").strip()
            if en or zh:
                out.append(f"- **{en}** — {zh}")
        out.append("")
    return "\n".join(out)


def _mmss(stamp: str) -> int:
    try:
        parts = [int(x) for x in str(stamp).replace("：", ":").split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        pass
    return 0


# --------------------------------------------------------------------------- #
# 悬浮窗
# --------------------------------------------------------------------------- #
class Bridge(QObject):
    asr_done = Signal(str)
    mt_done = Signal(str, str)
    quit_now = Signal()
    shot = Signal()
    summary_done = Signal(str)
    feed_done = Signal()
    level = Signal(float)
    speaking = Signal(bool)
    draft = Signal(str, str)
    silence = Signal(bool)


class Overlay(QWidget):
    def __init__(self):
        super().__init__(
            None,
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumSize(MIN_W, MIN_H)
        self.setMouseTracking(True)
        self.setWindowTitle("课堂同传")

        self._drag: QPoint | None = None
        self._mode: set[str] = set()
        self._start_geo: QRect | None = None
        self._start_pos: QPoint | None = None
        self._clickthrough = False
        self.size_locked = False
        self.font_scale = 1.0
        self.max_lines = 3
        self.opacity = 100
        self.draft_zh = True
        self.scale = 1.0
        self.theme = "dark"
        self.before_close = None
        self._speaking = False
        self._pulse_dim = False
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._on_pulse)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame(self)
        self.card.setObjectName("card")
        self.card.setMouseTracking(True)   # 没这个，悬停边缘不会触发鼠标样式变化
        outer.addWidget(self.card)

        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(20, 11, 18, 16)
        lay.setSpacing(9)

        bar = QHBoxLayout()
        bar.setSpacing(2)
        bar.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel(self.card)
        self.status.setObjectName("status")
        self.status.setMouseTracking(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status.setMinimumWidth(40)
        self.status.setTextFormat(Qt.RichText)
        bar.addWidget(self.status, 1)

        self.meter = LevelMeter(parent=self.card)
        bar.addWidget(self.meter, 0)

        self.btnbox = QWidget(self.card)
        bhb = QHBoxLayout(self.btnbox)
        bhb.setContentsMargins(8, 0, 0, 0)
        bhb.setSpacing(2)

        self.btn_main = QPushButton("▶ 开始", self.btnbox)
        self.btn_main.setObjectName("main")
        self.btn_main.setMinimumWidth(76)
        self.btn_main.setFixedHeight(22)
        self.btn_pause2 = QPushButton("暂停", self.btnbox)
        self.btn_pause2.setObjectName("tool")
        self.btn_pause2.setFixedSize(42, 20)
        self.btn_pause2.setCursor(Qt.PointingHandCursor)
        self.btn_pause2.setVisible(False)
        self.btn_gear = QPushButton("设置", self.btnbox)
        self.btn_close = QPushButton("✕", self.btnbox)
        self.btn_gear.setFixedSize(42, 20)
        self.btn_close.setFixedSize(24, 20)
        for b in (self.btn_main, self.btn_pause2, self.btn_gear, self.btn_close):
            b.setCursor(Qt.PointingHandCursor)
            bhb.addWidget(b)
        bar.addWidget(self.btnbox, 0)
        self.btnbox.setVisible(True)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(lambda: self.btnbox.setVisible(False))

        lay.addLayout(bar)

        line = QFrame(self.card)
        line.setObjectName("sep")
        line.setFixedHeight(1)
        lay.addWidget(line)

        self._entries: list[tuple[QLabel, str, str]] = []

        self.scroll = QScrollArea(self.card)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollBar:vertical{width:5px;background:transparent;}"
            "QScrollBar::handle:vertical{background:rgba(255,255,255,45);"
            "border-radius:2px;min-height:26px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
            "QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent;}"
        )
        self.scroll.viewport().setStyleSheet("background:transparent;")

        self.holder = QWidget()
        self.holder.setStyleSheet("background:transparent;")
        self.vbox = QVBoxLayout(self.holder)
        self.vbox.setContentsMargins(0, 6, 0, 2)
        self.vbox.setSpacing(14)
        self.vbox.addStretch(1)
        self.scroll.setWidget(self.holder)
        lay.addWidget(self.scroll, 1)

        self.draft = QLabel(self.card)
        self.draft.setObjectName("draft")
        self.draft.setWordWrap(True)
        self.draft.setVisible(False)
        lay.addWidget(self.draft)

        self.fade = FadeOut(QColor(22, 24, 30), self.scroll.viewport())

        self._placeholder: QLabel | None = QLabel(self.holder)
        self._placeholder.setTextFormat(Qt.RichText)
        self._placeholder.setText('<span style="color:#5E6B80">等待声音…</span>')
        self._placeholder.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.vbox.addWidget(self._placeholder)

        self._restyle()
        self.set_status("正在加载模型…", "#94a3b8")
        self._load_geometry()
        for w in (self.card, self.status, self.scroll.viewport()):
            w.installEventFilter(self)
        self.btn_close.clicked.connect(self.close)

    # -- 拖动 / 缩放 --------------------------------------------------------
    def eventFilter(self, obj, ev):
        et = ev.type()
        if et == QEvent.Resize and obj is self.scroll.viewport():
            self._place_fade()
            return False

        if et == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            gp = ev.globalPosition().toPoint()
            edges = self._edges_at(self.mapFromGlobal(gp))
            if edges:
                self._mode = edges
                self._start_geo = self.geometry()
                self._start_pos = gp
                self._drag = None
            else:
                self._mode = set()
                self._start_geo = None
                self._drag = gp - self.frameGeometry().topLeft()
            return True

        if et == QEvent.MouseMove:
            self._poke()
            gp = ev.globalPosition().toPoint()
            if self._mode:
                self._resize_to(gp)
                return True
            if self._drag is not None and (ev.buttons() & Qt.LeftButton):
                self.move(gp - self._drag)
                return True
            self.setCursor(self._cursor_for(self._edges_at(self.mapFromGlobal(gp))))
            return False

        if et == QEvent.Wheel and (ev.modifiers() & Qt.ControlModifier):
            self.scale = max(0.7, min(2.2, self.scale + (0.08 if ev.angleDelta().y() > 0 else -0.08)))
            self._restyle()
            self._save_geometry()
            return True

        if et == QEvent.MouseButtonRelease:
            self._mode = set()
            self._drag = None
            self._start_geo = None
            self._start_pos = None
            self._save_geometry()
            return True

        return super().eventFilter(obj, ev)

    # -- 外观 ---------------------------------------------------------------
    def _fade_qcolor(self) -> "QColor":
        op = max(40, min(100, int(getattr(self, "opacity", 100))))
        a = int(255 * op / 100)
        if os.environ.get("DSH_PALETTE") == "navy":
            return QColor(23, 30, 43, a)
        if self.theme == "light":
            return QColor(255, 255, 255, a)
        return QColor(22, 24, 30, a)

    def _palette(self) -> dict[str, str]:
        if os.environ.get("DSH_PALETTE") == "navy":
            return {
                "card": "rgba(23,30,43,251)",
                "border": "rgba(255,255,255,24)",
                "status": "#8FA6C2",
                "en": "#8CA3C0",
                "zh": "#F2F6FC",
                "tool": "rgba(198,212,230,165)",
                "hover": "rgba(255,255,255,26)",
                "accent": "#4ade80",
                "sep": "rgba(255,255,255,18)",
                "shadow": "26,32,48",
                "shadow_a": "110",
                "primary": "#3B6BF0",
                "primary_hover": "#2F5CDE",
                "draft": "#7E91AC",
            }
        if self.theme == "light":
            return {
                "card": "rgba(255,255,255,253)",
                "border": "rgba(229,232,239,255)",
                "status": "#98A0AF",
                "en": "#5F6878",
                "zh": "#161A22",
                "tool": "#98A0AF",
                "hover": "rgba(22,26,34,14)",
                "accent": "#0F9D58",
                "sep": "rgba(229,232,239,255)",
                "shadow": "26,32,48",
                "shadow_a": "60",
                "primary": "#3B6BF0",
                "primary_hover": "#2F5CDE",
                "draft": "#9AA3B2",
            }
        return {
            "card": "rgba(22,24,30,251)",
            "border": "rgba(255,255,255,22)",
            "status": "#8FA0B8",
            "en": "#8C9CB4",
            "zh": "#F2F5FA",
            "tool": "rgba(200,210,225,160)",
            "hover": "rgba(255,255,255,26)",
            "accent": "#4ade80",
            "sep": "rgba(255,255,255,16)",
            "shadow": "0,0,0",
            "shadow_a": "110",
            "primary": "#3B6BF0",
            "primary_hover": "#2F5CDE",
            "draft": "#7C8CA3",
        }

    def _restyle(self):
        s = self.scale
        c = self._palette()
        op = max(40, min(100, int(getattr(self, "opacity", 100))))
        alpha = int(255 * op / 100)
        c["card"] = re.sub(r",\s*\d+\)$", f",{alpha})", c["card"])
        self.setStyleSheet(
            f"""
            #card {{
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 16px;
            }}
            QLabel {{ font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; }}
            #status {{ color: {c['status']}; font-size: {11*s:.0f}px; }}
            #sep    {{ background-color: {c['sep']}; }}
            #draft  {{ color: {c['draft']}; font-size: {17*s:.0f}px; }}
            #tool   {{
                color: {c['tool']};
                background: transparent;
                border: none;
                font-size: 14px;
            }}
            #tool:hover {{ color: {c['zh']}; background: {c['hover']}; border-radius: 5px; }}
            #main {{
                color: #ffffff;
                background: {c['primary']};
                border: none;
                border-radius: 11px;
                font-size: 12px;
                padding: 0 12px;
            }}
            #main:hover {{ background: {c['primary_hover']}; }}
            """
        )
        if getattr(self, "_shadow", None) is not None:
            r, g, b = (int(x) for x in c["shadow"].split(","))
            self._shadow.setColor(QColor(r, g, b, int(c["shadow_a"])))
        if getattr(self, "meter", None) is not None:
            self.meter.set_color(c["accent"])
        if getattr(self, "fade", None) is not None:
            self.fade.set_color(self._fade_qcolor())
        for lbl, en, zh in getattr(self, "_entries", []):
            lbl.setText(self._html(en, zh))
        if getattr(self, "_entries", None):
            self._refresh_depths()

    def set_theme(self, theme: str):
        self.theme = "light" if theme == "light" else "dark"
        self._restyle()

    # -- 布局 ---------------------------------------------------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_fade()
        # 窗口窄了就把音量条收起来、状态文字换短版，避免控件挤在一起
        if getattr(self, "meter", None) is not None:
            self.meter.setVisible(self.width() >= 620)
        self._render_status()

    def _place_fade(self):
        if getattr(self, "fade", None) is None or getattr(self, "scroll", None) is None:
            return
        vp = self.scroll.viewport()
        self.fade.setGeometry(0, 0, vp.width(), 30)
        self.fade.raise_()

    # -- 位置记忆 -----------------------------------------------------------
    def _load_geometry(self):
        try:
            with open(UI_STATE, "r", encoding="utf-8-sig") as f:
                st = json.load(f)
            self.setGeometry(st["x"], st["y"], st["w"], st["h"])
            self.scale = float(st.get("scale", 1.0))
            self._restyle()
        except Exception:
            g = QApplication.primaryScreen().availableGeometry()
            self.setGeometry(g.center().x() - 380, g.bottom() - 230, 760, 190)

    def _save_geometry(self, force: bool = False):
        if getattr(self, "size_locked", False) and not force:
            return
        g = self.geometry()
        try:
            with open(UI_STATE, "w", encoding="utf-8") as f:
                json.dump({"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height(),
                           "scale": self.scale}, f)
        except Exception:
            pass

    def closeEvent(self, e):
        self._save_geometry()
        if self.before_close is not None and not self.before_close():
            e.ignore()
            return
        super().closeEvent(e)
        QApplication.quit()

    # -- 边缘缩放 -----------------------------------------------------------
    def _edges_at(self, p: QPoint) -> set[str]:
        if getattr(self, "size_locked", False):
            return set()
        x, y, w, h = p.x(), p.y(), self.width(), self.height()
        e: set[str] = set()
        if x <= EDGE:
            e.add("l")
        elif x >= w - EDGE:
            e.add("r")
        if y <= EDGE:
            e.add("t")
        elif y >= h - EDGE:
            e.add("b")
        return e

    @staticmethod
    def _cursor_for(e: set[str]) -> Qt.CursorShape:
        if e == {"l"} or e == {"r"}:
            return Qt.SizeHorCursor
        if e == {"t"} or e == {"b"}:
            return Qt.SizeVerCursor
        if e in ({"l", "t"}, {"r", "b"}):
            return Qt.SizeFDiagCursor
        if e in ({"r", "t"}, {"l", "b"}):
            return Qt.SizeBDiagCursor
        return Qt.ArrowCursor

    def _resize_to(self, gp: QPoint):
        if not self._start_geo or not self._start_pos:
            return
        dx = gp.x() - self._start_pos.x()
        dy = gp.y() - self._start_pos.y()
        g = QRect(self._start_geo)

        if "l" in self._mode:
            g.setLeft(min(g.left() + dx, g.right() - MIN_W))
        if "r" in self._mode:
            g.setRight(max(g.right() + dx, g.left() + MIN_W))
        if "t" in self._mode:
            g.setTop(min(g.top() + dy, g.bottom() - MIN_H))
        if "b" in self._mode:
            g.setBottom(max(g.bottom() + dy, g.top() + MIN_H))
        self.setGeometry(g)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        pos = e.position().toPoint()
        edges = self._edges_at(pos)
        if edges:
            self._mode = edges
            self._start_geo = self.geometry()
            self._start_pos = e.globalPosition().toPoint()
        else:
            self._mode = set()
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        gp = e.globalPosition().toPoint()
        if self._mode:
            self._resize_to(gp)
            return
        if self._drag is not None and e.buttons() & Qt.LeftButton:
            self.move(gp - self._drag)
            return
        self.setCursor(self._cursor_for(self._edges_at(e.position().toPoint())))

    def mouseReleaseEvent(self, e):
        self._mode = set()
        self._drag = None
        self._start_geo = None
        self._start_pos = None
        self._save_geometry()

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            self.scale = max(0.7, min(2.2, self.scale + (0.08 if e.angleDelta().y() > 0 else -0.08)))
            self._restyle()
            self._save_geometry()

    # -- 按钮 ---------------------------------------------------------------
    def _poke(self):
        pass

    def set_running(self, running: bool):
        self._running = running
        self.btn_main.setText("■ 停止" if running else "▶ 开始")
        self.btn_pause2.setVisible(running)
        if not running:
            self.btn_pause2.setText("暂停")

    def toggle_clickthrough(self):
        flags = self.windowFlags()
        if self._clickthrough:
            self.setWindowFlags(flags & ~Qt.WindowTransparentForInput)
        else:
            self.setWindowFlags(flags | Qt.WindowTransparentForInput)
        self._clickthrough = not self._clickthrough
        self.show()

    def set_status(self, text: str, color: str = "#4ade80"):
        self._status_full = text
        self._status_color = color
        self._render_status()

    def _render_status(self):
        full = getattr(self, "_status_full", "")
        color = getattr(self, "_status_color", "#4ade80")
        short = full.split(" · ")[-1] if " · " in full else full
        text = full if self.width() >= 560 else short
        if getattr(self, "_speaking", False) and getattr(self, "_pulse_dim", False):
            c = QColor(color)
            color = QColor(c.red(), c.green(), c.blue(), 90).name(QColor.HexArgb)
        self.status.setText(
            f'<span style="color:{color}">&#9679;</span>&nbsp;&nbsp;{text}'
        )

    def set_clickthrough(self, on: bool):
        """锁定鼠标穿透：点不到窗口，只能用托盘菜单解除。"""
        self._clickthrough = on
        self.setWindowFlag(Qt.WindowTransparentForInput, on)
        self.show()

    def set_silence(self, on: bool):
        """长时间没有声音时给出明确提示，别让人以为程序卡了。"""
        if not getattr(self, "_running", False):
            return
        if on:
            self.set_status("麦克风几乎没声音，检查设备或换个麦克风", "#f87171")
        else:
            self.set_status("正在录音 · 已恢复", "#f87171")

    def set_speaking(self, on: bool):
        """有人说话时让状态点呼吸，表示"在听，字还在路上"。"""
        self._speaking = on
        if on:
            if not self._pulse_timer.isActive():
                self._pulse_timer.start(420)
        else:
            self._pulse_timer.stop()
            self._pulse_dim = False
            self._render_status()

    def _on_pulse(self):
        self._pulse_dim = not self._pulse_dim
        self._render_status()

    def _html(self, en: str, zh: str, depth: int = 0) -> str:
        s = self.scale * self.font_scale
        c = self._palette()
        if depth <= 0:
            zs, es, zp, ep = 19.0, 11.0, 1.0, 1.0
        elif depth == 1:
            zs, es, zp, ep = 15.0, 10.0, 0.62, 0.55
        else:
            zs, es, zp, ep = 13.0, 9.5, 0.34, 0.30
        return (
            f'<div style="color:{c["en"]};font-size:{es*s:.0f}px;'
            f'line-height:150%;opacity:{ep}">{html.escape(en)}</div>'
            f'<div style="color:{c["zh"]};font-size:{zs*s:.0f}px;'
            f'line-height:145%;margin-top:3px;opacity:{zp}">{html.escape(zh)}</div>'
        )

    def _refresh_depths(self):
        keep = max(1, int(getattr(self, "max_lines", 3)))
        tail = self._entries[-keep:]
        for lbl, _, _ in self._entries:
            if lbl not in [t[0] for t in tail]:
                lbl.setVisible(False)
        for k, (lbl, en, zh) in enumerate(tail):
            lbl.setVisible(True)
            lbl.setText(self._html(en, zh, len(tail) - 1 - k))

    def set_draft(self, en: str = "", zh: str = ""):
        """正在说的那一句：一块固定高度区域，整块替换，说完升格成定稿。"""
        en = (en or "").strip()
        zh = (zh or "").strip() if getattr(self, "draft_zh", False) else ""
        if not en and not zh:
            self.draft.setText("")
            self.draft.setVisible(False)
            self.draft.setFixedHeight(0)
            return
        parts = []
        if en:
            parts.append(f'<span style="font-size:12px;color:#9AA3B2">⋯ {html.escape(en)}</span>')
        if zh:
            parts.append(f'<span style="font-size:15px;color:#9AA3B2">⋯ {html.escape(zh)}</span>')
        self.draft.setText("<br>".join(parts))
        self.draft.setTextFormat(Qt.RichText)
        self.draft.setFixedHeight(58 if zh else 36)   # 固定高，绝不撑大压住正文
        self.draft.setVisible(True)

    def show_text(self, orig: str, trans: str):
        self.set_draft()
        sb = self.scroll.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 40

        if self._placeholder is not None:
            self._placeholder.setParent(None)
            self._placeholder.deleteLater()
            self._placeholder = None

        lbl = QLabel(self.holder)
        lbl.setTextFormat(Qt.RichText)
        lbl.setWordWrap(True)
        lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lbl.setText(self._html(orig, trans, 0))
        self.vbox.addWidget(lbl)
        self._entries.append((lbl, orig, trans))
        self._refresh_depths()

        while len(self._entries) > 300:
            old, _, _ = self._entries.pop(0)
            old.setParent(None)
            old.deleteLater()

        if at_bottom:
            QTimer.singleShot(80, self._scroll_bottom)

    def _scroll_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 设置窗口
# --------------------------------------------------------------------------- #
def _save_config(cfg: dict) -> None:
    with open(os.path.join(ROOT, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _active_glossary(cfg: dict) -> str:
    """当前课程那一份词表。没有分类时退回旧的单份 glossary。"""
    courses = cfg.get("courses") or []
    act = cfg.get("active_course") or ""
    for c in courses:
        if c.get("name") == act:
            return (c.get("glossary") or "").strip()
    return (cfg.get("glossary") or "").strip()


def _input_devices() -> list[tuple[object, str]]:
    out: list[tuple[object, str]] = [(None, "系统默认麦克风")]
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, d["name"]))
    except Exception:
        pass
    return out


class SettingsDialog(QDialog):
    TR_LABELS = ["DeepSeek（用 key，质量最好）", "MyMemory（免费，不用 key）",
                 "本地翻译（不用 key，离线）", "不翻译，只出英文"]
    TR_VALUES = ["deepseek", "mymemory", "local", "off"]
    MODELS = ["small", "medium", "large-v3-turbo"]

    def __init__(self, parent, cfg: dict):
        super().__init__(parent)
        self.setWindowTitle("课堂同传 · 设置")
        self.setWindowFlag(Qt.Tool, True)
        self.setMinimumSize(520, 300)
        self.resize(560, 380)
        self._cfg = dict(cfg)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(10)

        tabs = QTabWidget()

        # ---------- 一、识别与翻译 ----------
        p1 = QWidget()
        f1 = QFormLayout(p1)
        f1.setContentsMargins(16, 16, 16, 12)
        f1.setSpacing(12)
        f1.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.cb_tr = QComboBox()
        self.cb_tr.addItems(self.TR_LABELS)
        cur = self._cfg.get("translator", "local")
        if cur in self.TR_VALUES:
            self.cb_tr.setCurrentIndex(self.TR_VALUES.index(cur))
        f1.addRow("翻译方式", self.cb_tr)

        self.ed_key = QLineEdit(self._cfg.get("api_key", ""))
        self.ed_key.setEchoMode(QLineEdit.Password)
        self.ed_key.setPlaceholderText("sk-...（只有 DeepSeek 才用到，只存在本机 config.json）")
        f1.addRow("API Key", self.ed_key)

        def _key_changed(text: str):
            if text.strip() and self.cb_tr.currentIndex() == 0:
                self.cb_tr.setCurrentIndex(1)

        self.ed_key.textChanged.connect(_key_changed)

        self.cb_model = QComboBox()
        for _label, _val in [
            ("轻量 small · 约 460 MB · 快，准确率一般", "small"),
            ("标准 large-v3-turbo · 约 780 MB · 推荐", "large-v3-turbo"),
            ("多语言 large-v3 · 约 3 GB · 支持日韩等", "large-v3"),
        ]:
            self.cb_model.addItem(_label, _val)
        _cur = self._cfg.get("model", "large-v3-turbo")
        for _i in range(self.cb_model.count()):
            if self.cb_model.itemData(_i) == _cur:
                self.cb_model.setCurrentIndex(_i)
        f1.addRow("识别模型", self.cb_model)

        self.cb_lang = QComboBox()
        for _label, _val in [
            ("英语（推荐，最快最准）", "en"),
            ("自动检测", None),
            ("日语", "ja"),
            ("韩语", "ko"),
            ("西班牙语", "es"),
            ("法语", "fr"),
            ("德语", "de"),
        ]:
            self.cb_lang.addItem(_label, _val)
        _cl = self._cfg.get("language", "en")
        for _i in range(self.cb_lang.count()):
            if self.cb_lang.itemData(_i) == _cl:
                self.cb_lang.setCurrentIndex(_i)
        f1.addRow("识别语言", self.cb_lang)

        self.cb_engine = QComboBox()
        self.cb_engine.addItem("Whisper（有标点，推荐）", "whisper")
        self.cb_engine.addItem("Parakeet（快约 3.5 倍，标点由翻译补）", "parakeet")
        _eng = self._cfg.get("asr_engine", "whisper")
        for _i in range(self.cb_engine.count()):
            if self.cb_engine.itemData(_i) == _eng:
                self.cb_engine.setCurrentIndex(_i)
        f1.addRow("识别引擎", self.cb_engine)

        self.cb_mic = QComboBox()
        for idx, name in _input_devices():
            self.cb_mic.addItem(str(name), idx)
        dev = self._cfg.get("device")
        for i in range(self.cb_mic.count()):
            if self.cb_mic.itemData(i) == dev:
                self.cb_mic.setCurrentIndex(i)
                break
        f1.addRow("麦克风", self.cb_mic)

        self.cb_src = QComboBox()
        self.cb_src.addItem("麦克风（上课）", "mic")
        self.cb_src.addItem("系统声音（网课/视频）", "system")
        _as = self._cfg.get("audio_source", "mic")
        for _i in range(self.cb_src.count()):
            if self.cb_src.itemData(_i) == _as:
                self.cb_src.setCurrentIndex(_i)
        f1.addRow("音频来源", self.cb_src)

        tip1 = QLabel("换模型或麦克风后要重启程序才生效；翻译方式和 key 保存即生效。")
        tip1.setWordWrap(True)
        tip1.setStyleSheet("color:#7b8798; font-size:12px;")
        f1.addRow("", tip1)
        tabs.addTab(p1, "识别与翻译")

        # ---------- 二、字幕外观 ----------
        p2 = QWidget()
        f2 = QFormLayout(p2)
        f2.setContentsMargins(16, 16, 16, 12)
        f2.setSpacing(12)
        f2.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.cb_theme = QComboBox()
        self.cb_theme.addItems(["深色", "浅色"])
        if self._cfg.get("theme") == "light":
            self.cb_theme.setCurrentIndex(1)
        f2.addRow("界面主题", self.cb_theme)

        self.cb_font = QComboBox()
        self.cb_font.addItems(["小", "标准", "大", "特大"])
        _fs = float(self._cfg.get("font_scale", 1.0))
        self.cb_font.setCurrentIndex(min(3, max(0, round((_fs - 0.85) / 0.15))))
        f2.addRow("字幕字号", self.cb_font)

        self.cb_lines = QComboBox()
        self.cb_lines.addItems(["1 行", "2 行", "3 行", "4 行", "5 行"])
        self.cb_lines.setCurrentIndex(min(4, max(0, int(self._cfg.get("max_lines", 3)) - 1)))
        f2.addRow("显示行数", self.cb_lines)

        self.cb_opacity = QComboBox()
        self.cb_opacity.addItems(["70%", "80%", "90%", "100%"])
        self.cb_opacity.setCurrentIndex(
            min(3, max(0, {70: 0, 80: 1, 90: 2, 100: 3}.get(
                int(self._cfg.get("opacity", 100)), 3)))
        )
        f2.addRow("背景不透明度", self.cb_opacity)

        self.cb_lock = QCheckBox("锁定窗口大小（拖不动）")
        self.cb_lock.setChecked(bool(self._cfg.get("lock_size")))
        f2.addRow("窗口", self.cb_lock)

        btn_remember = QPushButton("把当前大小设为默认")
        btn_remember.clicked.connect(self._remember_size)
        f2.addRow("", btn_remember)
        tabs.addTab(p2, "字幕外观")

        # ---------- 三、术语表 ----------
        p3 = QWidget()
        v3 = QVBoxLayout(p3)
        v3.setContentsMargins(16, 14, 16, 12)
        v3.setSpacing(8)

        tip3 = QLabel("一行一条。左边是容易听错的，右边是想要的写法。")
        tip3.setWordWrap(True)
        tip3.setStyleSheet("color:#7b8798; font-size:12px;")
        v3.addWidget(tip3)

        self._courses = [dict(c) for c in (self._cfg.get("courses") or [])]
        if not self._courses:
            self._courses = [{"name": "默认课程",
                              "glossary": self._cfg.get("glossary", "")}]
        _act = self._cfg.get("active_course") or self._courses[0]["name"]

        crow = QHBoxLayout()
        crow.addWidget(QLabel("课程"))
        self.cb_course = QComboBox()
        for c in self._courses:
            self.cb_course.addItem(c["name"])
        idx = [c["name"] for c in self._courses].index(_act) if _act in [
            c["name"] for c in self._courses] else 0
        self.cb_course.setCurrentIndex(idx)
        crow.addWidget(self.cb_course, 1)
        b_new = QPushButton("新建")
        b_new.setFixedWidth(58)
        b_del = QPushButton("删除")
        b_del.setFixedWidth(58)
        crow.addWidget(b_new)
        crow.addWidget(b_del)
        v3.addLayout(crow)

        self.ed_gloss = QPlainTextEdit(self._courses[idx]["glossary"])
        self.ed_gloss.setPlaceholderText(
            "课程：DI11006 风能\n"
            "week six / 第六周\n"
            "lab two / 实验二\n"
            "turbine 风机  blade 叶片  nacelle 机舱"
        )
        v3.addWidget(self.ed_gloss, 1)
        tabs.addTab(p3, "术语表")

        self.cb_course.currentIndexChanged.connect(self._switch_course)
        b_new.clicked.connect(self._new_course)
        b_del.clicked.connect(self._del_course)

        lay.addWidget(tabs, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Save).setText("保存")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.setWindowIcon(QIcon(os.path.join(ROOT, "app.png")))
        self.cb_theme.currentIndexChanged.connect(self._apply_theme)
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet("" if self.cb_theme.currentIndex() == 1 else DARK_QSS)

    def values(self) -> dict:
        cfg = dict(self._cfg)
        cfg["translator"] = self.TR_VALUES[self.cb_tr.currentIndex()]
        cfg["api_key"] = self.ed_key.text().strip()
        cfg["model"] = self.cb_model.currentData() or "large-v3-turbo"
        cfg["language"] = self.cb_lang.currentData() or "en"
        cfg["asr_engine"] = self.cb_engine.currentData() or "whisper"
        cfg["device"] = self.cb_mic.currentData()
        cfg["audio_source"] = self.cb_src.currentData() or "mic"
        cfg["theme"] = "light" if self.cb_theme.currentIndex() == 1 else "dark"
        cfg["glossary"] = self.ed_gloss.toPlainText().strip()
        name = self.cb_course.currentText() or "默认课程"
        for c in self._courses:
            if c["name"] == name:
                c["glossary"] = self.ed_gloss.toPlainText().strip()
        if not self._courses:
            self._courses = [{"name": name, "glossary": cfg["glossary"]}]
        cfg["courses"] = self._courses
        cfg["active_course"] = name
        cfg["glossary"] = next(
            (c["glossary"] for c in self._courses if c["name"] == name), cfg["glossary"])
        cfg["lock_size"] = self.cb_lock.isChecked()
        cfg["font_scale"] = [0.85, 1.0, 1.15, 1.35][self.cb_font.currentIndex()]
        cfg["max_lines"] = self.cb_lines.currentIndex() + 1
        cfg["opacity"] = [70, 80, 90, 100][self.cb_opacity.currentIndex()]
        return cfg

    def _remember_size(self):
        w = self.parent()
        if w is None:
            return
        try:
            w._save_geometry(force=True)
            self.cb_lock.setChecked(True)
        except Exception:
            pass

    def _switch_course(self, idx: int):
        prev = getattr(self, "_cur_idx", None)
        if prev is not None and 0 <= prev < len(self._courses) and prev != idx:
            self._courses[prev]["glossary"] = self.ed_gloss.toPlainText()
        if 0 <= idx < len(self._courses):
            self._cur_idx = idx
            self.ed_gloss.setPlainText(self._courses[idx]["glossary"])

    def _new_course(self):
        name, ok = QInputDialog.getText(self, "新建课程", "课程名称（例如：机械设计）")
        name = (name or "").strip()
        if not ok or not name:
            return
        if any(c["name"] == name for c in self._courses):
            return
        prev = getattr(self, "_cur_idx", 0)
        self._courses[prev]["glossary"] = self.ed_gloss.toPlainText()
        self._courses.append({"name": name, "glossary": ""})
        self.cb_course.blockSignals(True)
        self.cb_course.addItem(name)
        self.cb_course.setCurrentIndex(len(self._courses) - 1)
        self.cb_course.blockSignals(False)
        self._cur_idx = len(self._courses) - 1
        self.ed_gloss.setPlainText("")

    def _del_course(self):
        if len(self._courses) <= 1:
            QMessageBox.information(self, "提示", "至少留一个课程")
            return
        idx = self.cb_course.currentIndex()
        name = self._courses[idx]["name"]
        if QMessageBox.question(self, "删除课程", f"删掉「{name}」和它的词表？") != QMessageBox.Yes:
            return
        self._courses.pop(idx)
        self.cb_course.blockSignals(True)
        self.cb_course.removeItem(idx)
        self.cb_course.setCurrentIndex(min(idx, len(self._courses) - 1))
        self.cb_course.blockSignals(False)
        self._cur_idx = self.cb_course.currentIndex()
        self.ed_gloss.setPlainText(self._courses[self._cur_idx]["glossary"])


_HALLUCINATIONS = {
    "thank you.",
    "thanks.",
    "thank you",
    "you",
    ".",
    "bye.",
    "please subscribe.",
    "thank you for watching.",
}


def _dedupe(text: str) -> str:
    """Whisper 偶尔会把同一句吐两遍，这里把连续重复的句子压掉。"""
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and p == out[-1]:
            continue
        out.append(p)
    res = " ".join(out)
    half = len(res) // 2
    if half > 8 and res[:half].strip() == res[half:].strip():
        res = res[:half].strip()
    return res


def _modal(dlg) -> int:
    """让对话框和字幕窗同一层级并强制置前，否则会被"总在最前"的字幕窗盖住。"""
    dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    dlg.setWindowFlag(Qt.WindowDoesNotAcceptFocus, False)
    dlg.raise_()
    dlg.activateWindow()
    return dlg.exec()


def _ask(parent, title: str, text: str, yes: str, no: str) -> bool:
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    b_yes = box.addButton(yes, QMessageBox.AcceptRole)
    box.addButton(no, QMessageBox.RejectRole)
    _modal(box)
    return box.clickedButton() is b_yes


def _clean_title(text: str) -> str:
    """从总结里取一个能当文件夹名的标题。"""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("#").strip()
        line = re.sub(r"^(标题|title)\s*[:：]\s*", "", line, flags=re.I).strip()
        line = re.sub(r"^[\d一二三四五六七八九十]+\s*[.、)．]\s*", "", line)
        line = line.strip("*").strip()
        if not line or "这节课讲了什么" in line or "关键概念" in line:
            continue
        line = re.sub(r"[\\/:*?\"<>|\r\n]", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        return line[:22].strip(" .-_、，,：:")
    return ""


def _rename_session(folder: str, title: str) -> str | None:
    if not title:
        return None
    base = os.path.dirname(folder)
    dest = os.path.join(base, title)
    if os.path.abspath(dest) == os.path.abspath(folder):
        return None
    n = 2
    while os.path.exists(dest):
        dest = os.path.join(base, f"{title} ({n})")
        n += 1
    try:
        os.rename(folder, dest)
        return dest
    except Exception:
        return None


class SummaryDialog(QDialog):
    def __init__(self, parent, title: str, body: str, folder: str,
                 transcript: str = "", key: str = ""):
        super().__init__(parent)
        self._key = key
        self._summary = body
        self._transcript = transcript
        self._folder = folder
        self._title = title
        self._playing = False
        self._busy = False
        self._answers: queue.Queue = queue.Queue()
        self.setWindowTitle("本次总结")
        self.setWindowFlag(Qt.Tool, True)   # 不占任务栏，避免出现 pythonw 图标
        self.setMinimumSize(620, 620)
        self.setWindowIcon(QIcon(os.path.join(ROOT, "app.png")))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(10)

        head = QLabel(title or "本次总结")
        f = head.font()
        f.setPointSize(13)
        f.setBold(True)
        head.setFont(f)
        head.setWordWrap(True)
        lay.addWidget(head)

        tabs = QTabWidget()
        self.vue = QTextBrowser()
        self.vue.setStyleSheet("font-size:15px;")
        self.vue.setMarkdown(body)
        self.vue.setOpenLinks(False)
        self.vue.anchorClicked.connect(self._play)
        tabs.addTab(self.vue, "纪要")

        self.vtx = QTextBrowser()
        self.vtx.setStyleSheet("font-size:14px;")
        self.vtx.setHtml(self._transcript_html(transcript or "（没有记录）"))
        self.vtx.setOpenLinks(False)
        self.vtx.anchorClicked.connect(self._play)
        tabs.addTab(self.vtx, "原文")
        lay.addWidget(tabs, 1)

        # 问答：像千问一样放在最下面，问过之后才展开记录区
        self.chat = QTextBrowser()
        self.chat.setVisible(False)
        self.chat.setFixedHeight(96)          # 固定小条，不跟布局抢空间
        self.chat.setStyleSheet("font-size:13px;")
        lay.addWidget(self.chat)

        row2 = QHBoxLayout()
        self.qin = QLineEdit()
        self.qin.setPlaceholderText("就这次记录提问，例如：作业截止是哪天？")
        b_send = QPushButton("发送")
        b_send.setFixedWidth(64)
        row2.addWidget(self.qin, 1)
        row2.addWidget(b_send)
        lay.addLayout(row2)

        b_send.clicked.connect(self._ask)
        self.qin.returnPressed.connect(self._ask)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(300)

        row = QHBoxLayout()
        row.addStretch(1)
        b_copy = QPushButton("复制")
        b_terms = QPushButton("术语存进词表")
        b_word = QPushButton("导出 Word")
        b_pdf = QPushButton("导出 PDF")
        b_open = QPushButton("打开文件夹")
        b_close = QPushButton("关闭")
        for b in (b_copy, b_terms, b_word, b_pdf, b_open, b_close):
            row.addWidget(b)
        lay.addLayout(row)

        b_copy.clicked.connect(lambda: QApplication.clipboard().setText(body))
        b_terms.clicked.connect(self._save_terms)
        b_word.clicked.connect(self._export_word)
        b_pdf.clicked.connect(self._export_pdf)
        b_open.clicked.connect(lambda: os.startfile(folder))
        b_close.clicked.connect(self.accept)

    # ---------- 术语入库 ----------
    def _save_terms(self):
        lines = []
        hit = False
        for line in (self._summary or "").splitlines():
            s = line.strip()
            if s.startswith("### "):
                hit = "术语" in s
                continue
            if hit and s.startswith("- "):
                clean = _strip_md(s[2:]).strip()
                if clean and len(clean) < 60:
                    lines.append(clean)
        if not lines:
            self.chat.setVisible(True)
            self.chat.append("<b>这次总结里没有术语</b><br>")
            return
        cfg = _load_config()
        act = cfg.get("active_course") or "默认课程"
        courses = cfg.get("courses") or []
        if courses:
            for c in courses:
                if c.get("name") == act:
                    old = (c.get("glossary") or "").strip()
                    break
            else:
                old = (cfg.get("glossary") or "").strip()
        else:
            old = (cfg.get("glossary") or "").strip()
        have = {ln.strip() for ln in old.splitlines()}
        added = [ln for ln in lines if ln not in have]
        if not added:
            self.chat.setVisible(True)
            self.chat.append("<b>这些术语词表里已经有了</b><br>")
            return
        merged = (old + "\n" + "\n".join(added)).strip()
        cfg["glossary"] = merged
        for c in courses:
            if c.get("name") == act:
                c["glossary"] = merged
        cfg["courses"] = courses
        _save_config(cfg)
        self.chat.setVisible(True)
        self.chat.append(f"<b>已加入词表</b>：{len(added)} 条，下次识别就按这个纠正<br>")

    # ---------- 导出 ----------
    def _export_word(self):
        from docx import Document

        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Word", os.path.join(self._folder, (self._title or "纪要") + ".docx"),
            "Word 文档 (*.docx)")
        if not path:
            return
        doc = Document()
        doc.add_heading(self._title or "纪要", level=0)
        for line in (self._summary or "").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("### "):
                doc.add_heading(_strip_md(s[4:]), level=2)
            elif s.startswith("## "):
                doc.add_heading(_strip_md(s[3:]), level=1)
            elif s.startswith("> "):
                doc.add_paragraph(_strip_md(s[2:]), style="Intense Quote")
            elif s.startswith("- "):
                doc.add_paragraph(_strip_md(s[2:]), style="List Bullet")
            else:
                doc.add_paragraph(_strip_md(s))
        doc.save(path)
        self.chat.setVisible(True)
        self.chat.append(f"<b>已导出</b>：{html.escape(path)}<br>")

    def _export_pdf(self):
        from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument

        path, _ = QFileDialog.getSaveFileName(
            self, "导出 PDF", os.path.join(self._folder, (self._title or "纪要") + ".pdf"),
            "PDF (*.pdf)")
        if not path:
            return
        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QPageSize.A4))
        writer.setResolution(150)
        doc = QTextDocument()
        doc.setMarkdown(f"# {self._title or '纪要'}\n\n" + (self._summary or ""))
        doc.print_(writer)
        self.chat.setVisible(True)
        self.chat.append(f"<b>已导出</b>：{html.escape(path)}<br>")

    # ---------- 点击回放 ----------
    def _transcript_html(self, text: str) -> str:
        out = []
        for line in text.splitlines():
            s = line.rstrip()
            if not s:
                out.append("<p></p>")
                continue
            esc = html.escape(s)

            def _wrap(m):
                return (f'<a href="play:{_mmss(m.group(1))}" style="text-decoration:none">'
                        f'▶ [{m.group(1)}]</a>')

            esc = re.sub(r"\[(\d{1,2}:\d{2})\]", _wrap, esc)
            esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
            out.append(f"<p>{esc}</p>")
        return "\n".join(out)

    def _play(self, url):
        try:
            sec = int(str(url.toString()).split(":")[1])
        except Exception:
            return
        self._stop_audio()
        self._playing = True
        threading.Thread(target=self._play_from, args=(sec,), daemon=True).start()

    def _stop_audio(self):
        self._playing = False
        try:
            sd.stop()
        except Exception:
            pass

    def _play_from(self, sec: int):
        try:
            import av
            import numpy as np

            src = None
            for name in ("audio.opus", "audio.wav"):
                p = os.path.join(self._folder, name)
                if os.path.exists(p):
                    src = p
                    break
            if src is None:
                return
            container = av.open(src)
            stream = container.streams.audio[0]
            container.seek(max(0, int(sec) - 1) * av.time_base, stream=stream)
            res = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SR)
            want = SR * 14
            got = []
            for frame in container.decode(stream):
                for f in res.resample(frame):
                    arr = f.to_ndarray()
                    got.append(arr.reshape(-1))
                    if sum(len(g) for g in got) >= want:
                        break
                if sum(len(g) for g in got) >= want:
                    break
            container.close()
            if not got or not self._playing:
                return
            audio = np.concatenate(got).astype("float32") / 32768.0
            sd.play(audio, SR)
            sd.wait()
        except Exception:
            pass

    def _ask(self):
        q = self.qin.text().strip()
        if not q or self._busy:
            return
        if not self._key:
            self.chat.append("**提示**：没有配置 API key，无法提问。\n")
            return
        self.qin.clear()
        self._busy = True
        self.chat.setVisible(True)
        self.chat.append(f"<b>你</b>：{html.escape(q)}<br>")

        def work():
            try:
                self._answers.put(llm_qa(self._key, self._transcript, self._summary, q))
            except Exception as exc:
                self._answers.put(f"[失败] {exc.__class__.__name__}")

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            a = self._answers.get_nowait()
        except queue.Empty:
            return
        self._busy = False
        self.chat.append(f"<b>助教</b>：{html.escape(a)}<br>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--file", default=None, help="用音频文件代替麦克风（测试）")
    ap.add_argument("--test-loop", action="store_true", help="自检：录 5 秒系统声音并报告音量")
    ap.add_argument("--auto-rec", action="store_true", help="自检：启动 2 秒后自动开始录音")
    ap.add_argument("--device", default=None, help="麦克风设备名或序号")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--translator", default=None, choices=["local", "mymemory", "deepseek", "off"])
    ap.add_argument("--key", default=None, help="翻译服务的 API key（也可以写在 config.json）")
    ap.add_argument("--preview", action="store_true", help="只显示窗口，不采集音频")
    ap.add_argument("--shot", default=None, help="预览时把窗口存成图片")
    ap.add_argument("--dialog-shot", default=None, help="把设置窗口存成图片（自检用）")
    ap.add_argument("--quit-after", type=int, default=0, help="N 秒后自动退出（自检用）")
    ap.add_argument("--auto-test", action="store_true", help="用文件跑完整流程并自动结束（自检用）")
    args = ap.parse_args()

    cfg = _load_config()
    args.model = args.model or cfg.get("model") or "large-v3-turbo"
    args.threads = args.threads or int(cfg.get("threads") or 8)
    args.device = args.device or cfg.get("device")
    args.prompt = args.prompt or cfg.get("prompt") or DEFAULT_PROMPT
    args.translator = args.translator or cfg.get("translator") or "local"
    args.key = args.key or cfg.get("api_key") or ""
    if args.translator == "deepseek" and not args.key:
        args.translator = "local"

    # 单实例：已经有一个在跑就退出，避免两个窗口抢麦克风
    if sys.platform == "win32":
        try:
            import ctypes

            _h = ctypes.windll.kernel32.CreateMutexW(None, False, "CaptiNote.SingleInstance")
            if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
                print("已有一个 CaptiNote 在运行，本实例退出", flush=True)
                return 0
        except Exception:
            pass

    app = QApplication(sys.argv)
    # 主面板是 Qt.Tool，关掉设置/总结对话框时不能被当成"最后一个窗口"而退出
    app.setQuitOnLastWindowClosed(False)
    win = Overlay()
    win.show()
    _cfg0 = _load_config()
    _mdir = os.path.join(ROOT, "models")
    _has_model = os.path.isdir(_mdir) and any(
        os.scandir(_mdir) for _ in [0]
    )
    if not _has_model:
        win.set_status("首次启动：正在下载识别模型约 1.6 GB，请耐心等待…", "#fbbf24")
    win.size_locked = bool(_cfg0.get("lock_size"))
    win.font_scale = float(_cfg0.get("font_scale", 1.0))
    win.max_lines = int(_cfg0.get("max_lines", 3))
    win.opacity = int(_cfg0.get("opacity", 100))
    win.set_theme(_cfg0.get("theme", "dark"))

    if args.dialog_shot:
        dlg = SettingsDialog(win, _load_config())
        dlg.show()

        def _grab_dialog():
            dlg.grab().save(args.dialog_shot)
            print("dialog shot saved", flush=True)
            app.quit()

        QTimer.singleShot(700, _grab_dialog)
        return app.exec()

    if args.preview:
        win.set_status("预览模式", "#94a3b8")
        win.show_text(
            "Today we are going to discuss the design of machine elements.",
            "今天我们要讨论机械零件的设计。",
        )
        win.show_text(
            "When two parts must fit together, we define a clearance or an interference.",
            "当两个零件需要配合时，我们定义间隙配合或过盈配合。",
        )
        win.show_text(
            "A clearance fit always leaves a small gap between the shaft and the hole.",
            "间隙配合总会在轴和孔之间留下一点小的空隙。",
        )
        if args.shot:
            def _grab():
                pm = win.grab()
                pm.save(args.shot)
                print("shot saved", args.shot, flush=True)
                app.quit()
            QTimer.singleShot(900, _grab)
        return app.exec()

    bridge = Bridge()
    _cfg1 = _load_config()
    rec = Recognizer(_resolve_model(args.model), args.threads, args.prompt,
                     language=_cfg1.get("language", "en"),
                     engine=_cfg1.get("asr_engine", "whisper"))
    if args.translator == "local":
        factory = lambda: LocalMarian()
    elif args.translator == "mymemory":
        factory = lambda: MyMemory()
    elif args.translator == "deepseek":
        factory = lambda: DeepSeek(
            args.key,
            glossary=_active_glossary(cfg),
            target=cfg.get("target_lang", "zh"),
        )
    else:
        factory = lambda: Translator()
    tr = TranslatorWorker(factory)

    # 草稿专用的本地翻译线程（不进 API、不花钱）
    draft_tr = TranslatorWorker(lambda: LocalMarian())

    def _on_draft(en: str, zh: str):
        state["drafts"] = state.get("drafts", 0) + 1
        if not zh:
            bridge.draft.emit(en, "")
            return
        now_t = time.time()
        if now_t - state.get("draft_zh_t", 0) < 0.6:
            bridge.draft.emit(en, state.get("last_zh", ""))
            return
        state["draft_zh_t"] = now_t
        state["last_zh"] = zh
        bridge.draft.emit(en, zh)
        if os.environ.get("DSH_DEBUG_DRAFT"):
            print(f"DRAFT: {en}\n   → {zh}", flush=True)
        bridge.draft.emit(en, zh)

    draft_tr.on_text = _on_draft

    state = {"paused": False, "loading": True}
    state["draft_tr"] = draft_tr
    state["draft"] = None

    def _shot():
        try:
            win.grab().save(os.path.join(ROOT, "work", "live.png"))
            print("shot saved", flush=True)
        except Exception as exc:
            print("shot failed", exc, flush=True)

    bridge.shot.connect(_shot)
    hits = {"n": 0}

    def _show(o: str, t: str):
        if _is_filler_only(o) and _is_filler_only(t):
            return
        print(f"EN: {o}\nZH: {t}\n", flush=True)
        win.show_text(o, t)
        if session is not None:
            session.write_line(o, t)
        hits["n"] += 1
        if hits["n"] == 3:
            QTimer.singleShot(250, _shot)

    bridge.asr_done.connect(tr.submit)
    bridge.mt_done.connect(_show)
    bridge.quit_now.connect(app.quit)
    rec.on_text = bridge.asr_done.emit
    tr.on_text = bridge.mt_done.emit

    def _asr_out(text: str):
        if _is_filler_only(text):
            return
        bridge.asr_done.emit(text)

    rec.on_text = _asr_out

    def check_ready():
        if state["loading"] and rec.model is not None:
            state["loading"] = False
            win.set_status(f"{args.model} · 监听中")
    timer = QTimer()
    timer.timeout.connect(check_ready)
    timer.start(500)

    if args.quit_after > 0:
        QTimer.singleShot(args.quit_after * 1000, app.quit)

    def toggle_pause():
        _toggle_rec()

    def open_settings():
        old = _load_config()
        dlg = SettingsDialog(win, old)
        if _modal(dlg) != QDialog.Accepted:
            return
        new = dlg.values()
        _save_config(new)
        win.set_theme(new.get("theme", "dark"))
        win.size_locked = bool(new.get("lock_size"))
        win.font_scale = float(new.get("font_scale", 1.0))
        win.max_lines = int(new.get("max_lines", 3))
        win.opacity = int(new.get("opacity", 100))
        win._restyle()
        win._refresh_depths()

        if new.get("translator") != old.get("translator") or new.get("api_key") != old.get("api_key"):
            t = new.get("translator")
            if t == "mymemory":
                tr.reload(lambda: MyMemory())
            elif t == "deepseek" and new.get("api_key"):
                tr.reload(
                    lambda: DeepSeek(new["api_key"], glossary=_active_glossary(new))
                )
            elif t == "local":
                tr.reload(lambda: LocalMarian())
            else:
                tr.reload(lambda: Translator())

        if new.get("model") != old.get("model") or new.get("device") != old.get("device"):
            if _ask(win, "需要重启", "识别模型或麦克风改了，要现在重启吗？", "现在重启", "稍后"):
                win._save_geometry()
                os.execl(sys.executable, sys.executable, *sys.argv)

    win.btn_gear.clicked.connect(open_settings)

    seg_holder = {"seg": Segmenter()}
    try:
        seg_holder["seg"] = VadSegmenter()
    except Exception:
        pass

    session: Session | None = None
    stream_holder: dict[str, object] = {"s": None}
    state.update(recording=False)

    def _stop_mic() -> None:
        stream_holder["loop"] = False
        s = stream_holder["s"]
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        stream_holder["s"] = None

    def _shutdown() -> None:
        _stop_mic()
        if session is not None:
            session.close()

    app.aboutToQuit.connect(_shutdown)

    holder = {"dir": None, "transcript": ""}

    def _start_rec() -> None:
        nonlocal session
        if state["recording"]:
            return
        session = Session()

        # 草稿链路：本地流式识别 + 本地翻译（不花钱、不上网）
        try:
            _cfg_s = _load_config()
            if state.get("draft") is None:
                state["draft"] = DraftStreamer()
                state["draft"].on_draft = state["draft_tr"].submit
            else:
                state["draft"].reset()
        except Exception:
            state["draft"] = None

        # 统一走 _open_mic：麦克风 / 系统声音都由它决定
        if not _open_mic():
            session.discard()
            session = None
            win.set_status("音频设备打不开", "#f87171")
            _log("打开音频设备失败")
            return
        state["recording"] = True
        state["paused"] = False
        state["last_loud"] = time.time()
        state["pos"] = 0.0
        state["drafts"] = 0
        state["last_zh"] = ""
        state["draft_zh_t"] = 0
        state["emitted"] = 0.0
        state["silent_flag"] = False
        state["ctx"] = None
        win.set_running(True)
        src = "系统声音" if _load_config().get("audio_source") == "system" else "麦克风"
        win.set_status(f"正在录音 · {src}", "#f87171")

    def _open_mic() -> bool:
        # 系统声音：用 loopback 抓电脑正在播放的音频
        if _load_config().get("audio_source", "mic") == "system":
            def _loop():
                try:
                    import soundcard as sc

                    spk = sc.default_speaker()
                    mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
                    with mic.recorder(samplerate=SR, channels=1, blocksize=BLOCK) as rec:
                        while stream_holder.get("loop", False):
                            data = rec.record(numframes=BLOCK)
                            if data is None or len(data) == 0:
                                continue
                            mono = data[:, 0] if data.ndim > 1 else data
                            on_audio(np.asarray(mono, dtype=np.float32))
                except Exception as exc:
                    _log(f"环回采集失败 {exc.__class__.__name__}: {exc}")

            stream_holder["loop"] = True
            t = threading.Thread(target=_loop, daemon=True)
            t.start()
            stream_holder["s"] = None
            stream_holder["loop_thread"] = t
            _log("音频来源=系统声音(环回)")
            return True

        def cb(indata, frames, t, status):
            on_audio(indata[:, 0].astype(np.float32) / 32768.0)

        try:
            s = sd.InputStream(
                samplerate=SR, blocksize=BLOCK, channels=1, dtype="int16",
                device=args.device, callback=cb,
            )
            s.start()
        except Exception:
            return False
        stream_holder["s"] = s
        _log("音频来源=麦克风")
        return True

    def _toggle_pause() -> None:
        if not state["recording"]:
            return
        paused = not state.get("paused")
        state["paused"] = paused
        if paused:
            _stop_mic()
            win.btn_pause2.setText("继续")
            win.set_status("已暂停（麦克风已释放）", "#fbbf24")
        else:
            if _open_mic():
                win.btn_pause2.setText("暂停")
                win.set_status("正在录音", "#f87171")
            else:
                state["paused"] = True
                win.set_status("麦克风打不开", "#f87171")

    def _finish(recorded: Session) -> None:
        key = _load_config().get("api_key", "").strip()
        holder["dir"] = recorded.dir
        _log(f"对账 草稿 {state.get('drafts', 0)} 条 / 定稿 {len([l for l in text.splitlines() if l.strip()])} 行")
        try:
            _gl = _active_glossary(_load_config())
            _terms = []
            for _line in _gl.splitlines():
                _line = _line.strip()
                if not _line or _line.startswith("课程"):
                    continue
                for _part in _line.replace("/", " ").replace("／", " ").split():
                    if (len(_part) >= 3 and _part.isascii()
                            and any(c.isalpha() for c in _part)
                            and _part.lower() not in _GLOSS_STOP):
                        _terms.append(_part.lower())
            _terms = sorted(set(_terms))
            _low = text.lower()
            _hit = [t for t in _terms if t in _low]
            _log(f"术语命中 {len(_hit)}/{len(_terms)}：{', '.join(_hit[:12])}")
        except Exception:
            pass
        text = recorded.read_text()
        holder["transcript"] = text
        if not key:
            win.set_status("记录已保存（没配 key，未生成总结）", "#fbbf24")
            return
        _finish_text(text, key)

    def _retranslate_all(text: str, key: str, folder: str) -> None:
        """下课后按全文语境把逐句译文重译一遍，写 transcript_zh.md。"""
        try:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            lines = [l for l in lines if not l.startswith("#")]
            if len(lines) < 3:
                return
            gl = _active_glossary(_load_config())
            out = []
            for i in range(0, len(lines), 20):
                batch = lines[i:i + 20]
                zh = llm_translate_batch(batch, key, gl)
                out.extend(zip(batch, zh))
                _log(f"重译进度 {len(out)}/{len(lines)}")
            target = os.path.join(folder or os.path.join(ROOT, "recordings"), "transcript_zh.md")
            with open(target, "w", encoding="utf-8") as f:
                f.write("# 课后重译（按全文语境）\n\n")
                for en, zh in out:
                    f.write(f"- {en}\n  {zh}\n")
            _log(f"课后重译完成 {len(out)} 行 → {os.path.basename(target)}")
        except Exception as exc:
            _log(f"课后重译失败 {exc.__class__.__name__}: {exc}")

    def _finish_text(text: str, key: str) -> None:
        win.set_status("正在生成总结…", "#94a3b8")
        threading.Thread(target=_retranslate_all,
                         args=(text, key, holder.get("dir")), daemon=True).start()

        def work():
            try:
                obj = llm_summary(text, key, _load_config().get("target_lang", "zh"))
                bridge.summary_done.emit("OK|" + json.dumps(obj, ensure_ascii=False))
            except Exception as exc:
                bridge.summary_done.emit("ERR|" + exc.__class__.__name__)

        threading.Thread(target=work, daemon=True).start()

    def _stop_rec(discard: bool) -> None:
        nonlocal session
        if not state["recording"]:
            return
        _stop_mic()
        state["recording"] = False
        win.set_running(False)
        win.set_speaking(False)
        win.set_draft()
        if state.get("draft") is not None:
            try:
                state["draft"].reset()
            except Exception:
                pass
        if session is not None:
            session.close()
            # 转码可能要几十秒，放后台，别卡住界面
            threading.Thread(target=session.compress, daemon=True).start()
        if discard:
            if session is not None:
                session.discard()
            session = None
            win.set_status("已丢弃这次录音", "#94a3b8")
            return
        if session is not None and len(session.read_text()) > 80:
            _finish(session)
        else:
            win.set_status("没听到内容，记录已保存", "#fbbf24")
        session = None

    def _ask_stop() -> None:
        if not state["recording"]:
            return
        box = QMessageBox(win)
        box.setWindowTitle("结束录音")
        box.setText("这次录音要怎么处理？")
        b_save = box.addButton("结束并总结", QMessageBox.AcceptRole)
        b_drop = box.addButton("丢弃这次", QMessageBox.DestructiveRole)
        box.addButton("继续录音", QMessageBox.RejectRole)
        _modal(box)
        hit = box.clickedButton()
        if hit is b_save:
            _stop_rec(False)
        elif hit is b_drop:
            if _ask(win, "确认丢弃", "这段录音会被删除，无法恢复。确定？", "丢弃", "取消"):
                _stop_rec(True)

    def _toggle_rec() -> None:
        if state["recording"]:
            _ask_stop()
        else:
            _start_rec()

    def _on_summary(msg: str):
        tag, body = msg.split("|", 1)
        if tag != "OK":
            win.set_status(f"总结失败：{body}", "#f87171")
            return
        try:
            obj = json.loads(body)
            md = summary_md(obj)
            title = (obj.get("title") or "").strip() or _clean_title(md)
        except Exception:
            md = body
            title = _clean_title(md)
        folder = holder["dir"] or ROOT
        try:
            with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as f:
                f.write(md)
        except Exception:
            pass
        moved = _rename_session(folder, title) if folder != ROOT else None
        if moved:
            folder = moved
            holder["dir"] = moved
        win.set_status("总结好了", "#4ade80")
        if state.get("auto_test"):
            print("SUMMARY_OK|" + folder, flush=True)
            QTimer.singleShot(400, app.quit)
            return
        _modal(SummaryDialog(win, title or "本次总结", md, folder,
                             holder.get("transcript", ""),
                             key=_load_config().get("api_key", "")))

    bridge.summary_done.connect(_on_summary)
    bridge.level.connect(win.meter.push)
    bridge.speaking.connect(win.set_speaking)
    bridge.draft.connect(win.set_draft)
    bridge.silence.connect(win.set_silence)

    def _model_watch():
        """模型没加载完时，把下载/加载进度显示出来，别让人以为卡死。"""
        loaded = (getattr(rec, "model", None) is not None) or (getattr(rec, "para", None) is not None)
        if loaded:
            if not state.get("model_ready"):
                state["model_ready"] = True
                _log("模型加载完成")
                if not state.get("recording"):
                    win.set_status("准备就绪 · 点「开始」", "#94a3b8")
            return
        if state.get("model_ready"):
            return
        try:
            total = 0
            mdir = os.path.join(ROOT, "models")
            for root, _dirs, files in os.walk(mdir):
                for fn in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
            mb = total // (1024 * 1024)
        except Exception:
            mb = 0
        if mb > 50:
            win.set_status(f"正在加载识别模型… 已就位 {mb} MB", "#fbbf24")
        else:
            win.set_status("首次启动：正在下载识别模型（约 1.6 GB），请保持联网…", "#fbbf24")

    _model_timer = QTimer()
    _model_timer.timeout.connect(_model_watch)
    _model_timer.start(1000)

    def _last_unfinished():
        base = os.path.join(ROOT, "recordings")
        best = None
        try:
            for d in os.listdir(base):
                p = os.path.join(base, d)
                md = os.path.join(p, "transcript.md")
                if (os.path.isdir(p) and os.path.exists(md)
                        and not os.path.exists(os.path.join(p, "summary.md"))
                        and os.path.getsize(md) > 400):
                    t = os.path.getmtime(md)
                    if best is None or t > best[0]:
                        best = (t, p)
        except Exception:
            return None
        return best[1] if best else None

    def _offer_recovery():
        p = _last_unfinished()
        if not p:
            return
        key = _load_config().get("api_key", "").strip()
        if not key:
            return
        if _ask(win, "补总结", f"发现一次没生成总结的记录：\n{os.path.basename(p)}\n要现在补吗？",
                "补总结", "跳过"):
            holder["dir"] = p
            try:
                with open(os.path.join(p, "transcript.md"), encoding="utf-8-sig") as f:
                    holder["transcript"] = f.read()
            except Exception:
                holder["transcript"] = ""
            _finish_text(holder["transcript"], key)

    QTimer.singleShot(2500, _offer_recovery)
    def _silence_watch():
        if not state.get("recording") or state.get("paused"):
            return
        silent_for = time.time() - state.get("last_loud", time.time())
        was = state.get("silent_flag", False)
        if silent_for > 6.0 and not was:
            state["silent_flag"] = True
            bridge.silence.emit(True)
            _log("静音超过6秒，提示用户检查设备")
        elif silent_for < 1.5 and was:
            state["silent_flag"] = False
            bridge.silence.emit(False)
            _log("声音恢复")

    _silence_timer = QTimer()
    _silence_timer.timeout.connect(_silence_watch)
    _silence_timer.start(2000)

    win.btn_main.clicked.connect(_toggle_rec)
    win.btn_pause2.clicked.connect(_toggle_pause)
    if getattr(args, "auto_rec", False):
        QTimer.singleShot(2000, win.btn_main.click)

    icon_png = os.path.join(ROOT, "app.png")
    app_icon = QIcon(icon_png) if os.path.exists(icon_png) else QIcon()
    app.setWindowIcon(app_icon)
    win.setWindowIcon(app_icon)

    tray = QSystemTrayIcon(app_icon, app)
    tray.setToolTip("课堂同传")
    menu = QMenu()
    act_toggle = menu.addAction("显示 / 隐藏字幕")
    act_pause = menu.addAction("开始 / 结束录音")
    menu.addSeparator()
    act_settings = menu.addAction("设置…")
    act_folder = menu.addAction("打开记录文件夹")
    act_import = menu.addAction("导入外部记录（AudioScript 等导出的 md）…")
    act_through = menu.addAction("鼠标穿透（锁定后只能用托盘解除）")
    act_through.setCheckable(True)
    menu.addSeparator()
    act_quit = menu.addAction("退出")
    tray.setContextMenu(menu)
    tray.show()

    act_toggle.triggered.connect(lambda: win.setVisible(not win.isVisible()))
    act_pause.triggered.connect(_toggle_rec)
    act_settings.triggered.connect(open_settings)
    act_folder.triggered.connect(
        lambda: os.startfile(holder["dir"] or os.path.join(ROOT, "recordings"))
    )

    def _import_record():
        """导入别的工具导出的逐字稿，直接生成纪要与术语。"""
        path, _ = QFileDialog.getOpenFileName(
            win, "选择导出的记录", "", "文本 (*.md *.txt);;所有文件 (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as f:
                text = f.read()
        except Exception as exc:
            win.set_status(f"读不了这个文件：{exc.__class__.__name__}", "#f87171")
            return
        if len(text.strip()) < 50:
            win.set_status("文件内容太少，没东西可总结", "#fbbf24")
            return
        stamp = datetime.datetime.now().strftime("%m%d_%H%M")
        folder = os.path.join(ROOT, "recordings", f"{stamp}_导入")
        try:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "transcript.md"), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            folder = os.path.dirname(path)
        holder["dir"] = folder
        holder["transcript"] = text
        key = _load_config().get("api_key", "").strip()
        if not key:
            win.set_status("没有 API key，无法生成纪要", "#fbbf24")
            return
        _log(f"导入外部记录 {os.path.basename(path)} {len(text)} 字")
        _finish_text(text, key)

    act_import.triggered.connect(_import_record)
    act_through.toggled.connect(win.set_clickthrough)
    act_quit.triggered.connect(app.quit)
    tray.activated.connect(
        lambda reason: win.setVisible(not win.isVisible())
        if reason == QSystemTrayIcon.Trigger
        else None
    )
    app.aboutToQuit.connect(tray.hide)

    def on_audio(audio: np.ndarray):
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float32)))) + 1e-9
        db = 20.0 * np.log10(rms)
        bridge.level.emit(max(0.0, min(1.0, (db + 55.0) / 40.0)))
        if session is not None:
            session.write_audio(audio)
        piece = seg_holder["seg"].feed(audio)
        state["pos"] = state.get("pos", 0.0) + len(audio) / SR
        if piece is not None and len(piece) > SR // 5:
            ctx = state.get("ctx")
            ctx_n = 0 if ctx is None else len(ctx)
            feed_len = (ctx_n + len(piece)) / SR
            window_start = max(0.0, state["pos"] - feed_len)
            skip = max(0.0, state.get("emitted", 0.0) - window_start)
            rec.submit(piece, ctx, skip)
            state["ctx"] = piece[-SR * 8:].copy()
            state["emitted"] = state["pos"]
        ds = state.get("draft")
        if ds is not None:
            try:
                ds.feed(audio)
            except Exception:
                state["draft"] = None
        spk = bool(getattr(seg_holder["seg"], "speaking", False))
        if spk != state.get("speaking"):
            state["speaking"] = spk
            bridge.speaking.emit(spk)

        # 记录最近一次听到声音的时间，判断交给看门狗定时器
        if db > -60.0:
            state["last_loud"] = time.time()

    def _confirm_quit() -> bool:
        if not state["recording"]:
            return True
        box = QMessageBox(win)
        box.setWindowTitle("正在录音")
        box.setText("还在录音，确定退出吗？")
        b_save = box.addButton("结束并保存", QMessageBox.AcceptRole)
        b_drop = box.addButton("丢弃这次", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        _modal(box)
        hit = box.clickedButton()
        if hit is b_save:
            _stop_rec(False)
            return True
        if hit is b_drop:
            _stop_rec(True)
            return True
        return False

    win.before_close = _confirm_quit

    def _on_feed_done():
        if state.get("auto_test"):
            _stop_rec(False)
        else:
            app.quit()

    bridge.feed_done.connect(_on_feed_done)

    if args.test_loop:
        import soundcard as sc

        try:
            spk = sc.default_speaker()
            print("默认扬声器:", spk.name, flush=True)
            mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
            print("环回设备:", mic.name if mic else "拿不到", flush=True)
            chunks = []
            with mic.recorder(samplerate=SR, channels=1, blocksize=1600) as r:
                for _ in range(50):
                    d = r.record(numframes=1600)
                    if d is not None and len(d):
                        chunks.append(d[:, 0].copy())
            if chunks:
                a = np.concatenate(chunks)
                rms = float(np.sqrt(np.mean(a.astype(np.float32) ** 2)))
                peak = float(np.max(np.abs(a)))
                print(f"采到 {len(a)/SR:.1f}s  RMS={rms:.4f}  峰值={peak:.3f}", flush=True)
                print("判定:", "有声音" if rms > 0.01 else "几乎静音（电脑可能没在放声音）", flush=True)
            else:
                print("一个样本都没采到", flush=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print("失败:", exc.__class__.__name__, exc, flush=True)
        return 0

    if args.file:
        state["auto_test"] = bool(args.auto_test)
        audio = decode_audio(args.file, sampling_rate=SR)
        win.set_status(f"文件模式：{os.path.basename(args.file)}")
        if args.auto_test:
            _start_rec()
        threading.Thread(
            target=_feed_file, args=(audio, on_audio, rec, tr, bridge, win), daemon=True
        ).start()
    else:
        win.set_status("准备就绪 · 点「开始」", "#94a3b8")

    return app.exec()


def _feed_file(audio, on_audio, rec, tr, bridge, win):
    step = BLOCK
    for i in range(0, len(audio), step):
        on_audio(audio[i: i + step])
        time.sleep(0.02)
    for _ in range(8):
        on_audio(np.zeros(step, dtype=np.float32))
        time.sleep(0.02)

    # 等模型加载 + 两个队列排空，最多 180 秒
    deadline = time.time() + 180
    while time.time() < deadline:
        if rec.model is not None and rec.jobs.empty() and tr.jobs.empty():
            time.sleep(2.0)
            break
        time.sleep(0.3)

    win.set_status("测试结束")
    print("done", flush=True)
    bridge.feed_done.emit()


if __name__ == "__main__":
    raise SystemExit(main())
