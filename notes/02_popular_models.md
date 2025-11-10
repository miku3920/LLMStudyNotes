# 常見的大型語言模型

## ChatGPT

### 定義
ChatGPT 是由 OpenAI 開發的對話式 AI 模型，基於 GPT（Generative Pre-trained Transformer）架構。

### 特點
- **版本**：主要有 GPT-3.5 和 GPT-4 兩個版本
- **能力**：自然語言對話、創意寫作、程式編寫、問題解答、翻譯等
- **訓練方法**：Pre-training + Instruction Fine-tuning + RLHF
- **發布時間**：2022 年 11 月推出，引發 AI 熱潮

### 應用場景
- 對話助理
- 內容創作
- 程式碼生成與除錯
- 教育輔助
- 資料分析協助

## Gemini

### 定義
Gemini 是 Google 開發的多模態大型語言模型，前身為 Bard。

### 特點
- **多模態能力**：原生支援文本、圖像、音訊、影片等多種資料類型
- **版本**：Gemini Nano、Gemini Pro、Gemini Ultra
- **整合**：深度整合到 Google 產品生態系統
- **發布時間**：2023 年推出

### 優勢
- 原生多模態理解（不需要額外模組）
- 與 Google 服務緊密整合
- 強大的推理能力

## BERT

### 定義
BERT（Bidirectional Encoder Representations from Transformers）是 Google 在 2018 年推出的預訓練語言模型。

### 架構特點
- **雙向編碼器**：同時考慮前後文脈絡
- **Transformer Encoder**：只使用 Transformer 的編碼器部分
- **訓練方法**：Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)

### 主要用途
- 句子分類
- 命名實體識別（NER）
- 問答系統
- 文本理解任務

### 影響
BERT 開啟了預訓練語言模型的新時代，後續衍生出許多變體如 RoBERTa、ALBERT、DistilBERT 等。

## T5

### 定義
T5（Text-To-Text Transfer Transformer）是 Google 在 2019 年推出的統一框架模型。

### 核心概念
將所有 NLP 任務統一為「文本到文本」的格式：
- 翻譯：`translate English to German: That is good. → Das ist gut.`
- 摘要：`summarize: [長文本] → [摘要]`
- 分類：`sentiment: This movie is great! → positive`

### 架構
- **Encoder-Decoder**：完整的 Transformer 架構
- **統一介面**：所有任務使用相同的輸入輸出格式

### 優勢
- 簡化了多任務學習
- 便於遷移學習
- 靈活的任務定義

## GPT

### 定義
GPT（Generative Pre-trained Transformer）是 OpenAI 開發的自回歸語言模型系列。

### 演進歷程
- **GPT-1**（2018）：1.17 億參數，證明預訓練的有效性
- **GPT-2**（2019）：15 億參數，展現驚人的生成能力
- **GPT-3**（2020）：1750 億參數，In-context Learning 能力
- **GPT-3.5**（2022）：ChatGPT 的基礎
- **GPT-4**（2023）：多模態能力，更強推理

### 特點
- **自回歸生成**：逐字預測下一個 Token
- **Decoder-only**：只使用 Transformer 的解碼器
- **In-context Learning**：可透過範例學習新任務（Few-shot Learning）

### 應用
- 文本生成
- 對話系統
- 程式碼生成
- 創意寫作

## LLaMA

### 定義
LLaMA（Large Language Model Meta AI）是 Meta（前 Facebook）在 2023 年推出的開源大型語言模型。

### 版本
- **LLaMA 1**：7B、13B、33B、65B 參數版本
- **LLaMA 2**：7B、13B、70B 參數版本，包含 Chat 版本
- **LLaMA 3**：進一步改進的版本

### 特點
- **開源**：推動了開源 LLM 社群的快速發展
- **效率**：在較小參數量下達到優秀性能
- **可微調**：適合進行客製化微調

### 影響
LLaMA 的開源催生了大量衍生模型，如 Alpaca、Vicuna、Llama-2-Chat 等，大幅降低了 LLM 應用的門檻。

## Gemma

### 定義
Gemma 是 Google DeepMind 開發的輕量級開源語言模型。

### 特點
- **開源**：完全開源，可商用
- **輕量級**：2B 和 7B 參數版本
- **高效能**：在相對較小的模型規模下提供優秀性能
- **安全**：經過嚴格的安全性訓練

### 定位
Gemma 定位於提供易於部署和微調的開源替代方案，適合資源受限的場景。

## 模型選擇建議

### 任務類型
- **理解任務**（分類、NER）：BERT 系列
- **生成任務**（寫作、對話）：GPT、LLaMA
- **統一框架**：T5
- **對話應用**：ChatGPT、Gemini、Llama-2-Chat

### 資源考量
- **高性能需求**：GPT-4、Gemini Ultra
- **平衡性能與成本**：GPT-3.5、Gemini Pro
- **本地部署**：LLaMA、Gemma

### 開源需求
- **開源可微調**：LLaMA、Gemma
- **商業產品**：ChatGPT、Gemini

