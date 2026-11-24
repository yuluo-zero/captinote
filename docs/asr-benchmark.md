# 识别引擎实测对比（2026-11-25）

测试机器：AMD Ryzen AI 9 HX 470（12 核 24 线程）、31 GB 内存、无 NVIDIA 显卡。
两个模型都跑 CPU，8 线程。

## 速度

| 模型 | 5 分钟音频耗时 | 实时倍率 |
|---|---:|---:|
| Whisper large-v3-turbo (int8) | 61.6 s | 4.87× |
| Parakeet TDT 0.6B v2 (int8) | 17.6 s | **17.0×** |

Parakeet 大约快 **3.5 倍**。按我们每段 5 秒的节奏推算，单段识别可从约 1.0 秒降到 0.3 秒。

## 准确率

### 干净音频（60 秒讲课片段）

两者内容几乎一致：

- Whisper：`for a lot of times already, but we haven't actually had a class on it. So today we're going to do that in week six, lab two.`
- Parakeet：`for a lot of times already but we haven't actually uh had a class on it um so today we're going to do that in week six lab two`

差别：Whisper 有标点；Parakeet 保留 uh/um 填充词、无标点。

### 远场课堂录音（5 分钟，手机放在教室）

两者都不理想，互相 WER ≈ 31.8%：

- Whisper：`Next day you bring this truck. Then what I do is this one.`
- Parakeet：`Lastly, you read this trap. Then what I do is this one you guys give me this trap`

这一段老师真正说的是"明天把报告带来"。**两个模型都听错了同一个词**，说明瓶颈在录音质量，不在模型。

## 结论

1. **准确率：干净音频下两者相当；远场录音下两者同样不行。** 换模型救不了麦克风。
2. **速度：Parakeet 快 3.5 倍**，如果延迟是主要痛点，它值得接。
3. **代价：Parakeet 输出无标点、带填充词**，可读性和翻译质量可能略降（中文标点由翻译模型补）。

## 建议

- 短期：继续用 Whisper（有标点、可给 initial_prompt 喂课程词表）
- 中期：把 Parakeet 做成可选引擎，用真实课堂录音再比一次，由使用者自己选"快"还是"准"
- 长期：真正的提升来自**麦克风**，不是模型
