# Token 與 Tokenization（分詞）

## Token 的定義

### 什麼是 Token？
Token 是自然語言處理（NLP）中文本的最小處理單位。在大型語言模型中，文本必須先被切分成 Token，才能被模型處理。

### Token 的類型
Token 可以是：
- **單詞**：`"hello"`、`"world"`
- **子詞**（Subword）：`"playing"` → `["play", "ing"]`
- **字元**：`"a"`、`"b"`、`"中"`
- **特殊符號**：標點符號、空格、換行

### 為什麼需要 Token？
1. **統一輸入格式**：將不同長度的文本轉換為固定的數字序列
2. **詞彙表管理**：避免詞彙表過大（如果每個可能的詞都要一個 ID，詞彙表會非常龐大）
3. **處理未知詞**：透過子詞切分，可以處理訓練時未見過的詞

## Tokenization（分詞）方法

### 1. Word-level Tokenization（詞級分詞）

**方法**：將文本按單詞切分。

**範例**：
```
輸入："I love natural language processing"
輸出：["I", "love", "natural", "language", "processing"]
```

**優點**：
- 直觀、易理解
- 保留完整的語義單位

**缺點**：
- 詞彙表會非常大
- 無法處理未知詞（OOV, Out of Vocabulary）
- 對於中文等語言需要額外的分詞工具

### 2. Character-level Tokenization（字元級分詞）

**方法**：將文本按字元切分。

**範例**：
```
輸入："hello"
輸出：["h", "e", "l", "l", "o"]
```

**優點**：
- 詞彙表非常小（只需要字元集的大小）
- 沒有未知詞問題

**缺點**：
- 序列長度大幅增加
- 難以學習詞彙層級的語義
- 計算成本高

### 3. Subword Tokenization（子詞分詞）

**方法**：將詞切分成更小的有意義單位。這是目前主流的方法。

#### BPE (Byte Pair Encoding)

**原理**：
1. 從字元級開始
2. 反覆合併最常見的相鄰 Token 對
3. 建立固定大小的詞彙表

**範例**：
```
原始詞彙："low", "lower", "newest", "widest"
字元級："l o w", "l o w e r", "n e w e s t", "w i d e s t"

迭代合併：
- 合併 "e" + "s" → "es"
- 合併 "es" + "t" → "est"
- 合併 "l" + "o" → "lo"
- ...

最終："lo w", "lo w er", "new est", "wid est"
```

**使用模型**：GPT、GPT-2

#### WordPiece

**原理**：
類似 BPE，但選擇合併時考慮語言模型的可能性，而不只是頻率。

**特點**：
- 子詞通常以 `##` 開頭表示不是詞的開頭
- `"playing"` → `["play", "##ing"]`

**使用模型**：BERT

#### SentencePiece

**原理**：
- 將文本視為 Unicode 字元序列
- 不依賴語言特定的預處理（如分詞）
- 支援 BPE 和 Unigram Language Model

**特點**：
- 語言無關
- 將空格也視為特殊字元（用 ▁ 表示）
- `"hello world"` → `["▁hello", "▁world"]`

**使用模型**：T5、LLaMA、Gemma

### 4. 各方法比較

| 方法 | 詞彙表大小 | 序列長度 | 未知詞處理 | 代表模型 |
|------|-----------|----------|-----------|---------|
| Word-level | 非常大 | 短 | 差 | 早期模型 |
| Character-level | 非常小 | 非常長 | 完美 | 少數模型 |
| BPE | 中等 | 中等 | 好 | GPT |
| WordPiece | 中等 | 中等 | 好 | BERT |
| SentencePiece | 中等 | 中等 | 好 | T5, LLaMA |

## Token 的實際應用

### Token 計數與成本

API 服務（如 OpenAI）通常按 Token 數量收費：

```python
import tiktoken

# 使用 GPT-4 的 tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")

text = "Hello, how are you today?"
tokens = encoding.encode(text)

print(f"文本: {text}")
print(f"Token 數量: {len(tokens)}")
print(f"Tokens: {tokens}")
```

**輸出**：
```
文本: Hello, how are you today?
Token 數量: 7
Tokens: [9906, 11, 1268, 527, 499, 3432, 30]
```

### 不同語言的 Token 效率

#### 英文
```
"Hello world" → 2 tokens
```

#### 中文
```
"你好世界" → 4 tokens (每個字通常是 1-2 個 tokens)
```

**觀察**：中文通常比英文需要更多 Token，因為訓練資料以英文為主。

### Context Window（上下文視窗）

模型的上下文視窗以 Token 數量計算：

| 模型 | Context Window |
|------|----------------|
| GPT-3.5-turbo | 4,096 tokens |
| GPT-4 | 8,192 tokens |
| GPT-4-32k | 32,768 tokens |
| GPT-4-turbo | 128,000 tokens |
| Claude 3 | 200,000 tokens |

**注意事項**：
- 輸入和輸出的 Token 都計入上下文視窗
- 超過限制會被截斷或拒絕

## Tokenization 實作範例

### 使用 Hugging Face Transformers

```python
from transformers import AutoTokenizer

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization is important!"

# 編碼
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.encode(text)

print(f"Tokens: {tokens}")
print(f"Input IDs: {input_ids}")

# 解碼
decoded = tokenizer.decode(input_ids)
print(f"Decoded: {decoded}")
```

**輸出**：
```
Tokens: ['token', '##ization', 'is', 'important', '!']
Input IDs: [101, 19204, 3989, 2003, 2590, 999, 102]
Decoded: [CLS] tokenization is important! [SEP]
```

### 特殊 Token

模型通常使用特殊 Token：

- **[CLS]**：句子開始（BERT）
- **[SEP]**：句子分隔
- **[PAD]**：填充到固定長度
- **[UNK]**：未知詞
- **[MASK]**：遮罩（用於訓練）
- **<|endoftext|>**：文本結束（GPT）

## Token 的挑戰與限制

### 1. 語言不平等
- 英文通常更高效（1 個詞 ≈ 1 個 Token）
- 中文、日文等需要更多 Token（1 個字 ≈ 1-2 個 Token）
- 導致成本和效能差異

### 2. 詞彙表固定
- 新詞、專有名詞可能被切得很碎
- 難以處理特定領域術語

### 3. Token 邊界問題
```python
# 範例
"ChatGPT" → ["Chat", "G", "PT"]  # 可能被意外切分
"COVID-19" → ["CO", "VI", "D", "-", "19"]  # 數字和符號處理
```

### 4. Context Window 限制
長文本處理的挑戰：
- 需要切分文本
- 可能遺失跨段落的上下文
- 需要策略如滑動視窗、摘要等

## 最佳化 Token 使用

### 1. 精簡提示詞
```python
# 冗長版本 (更多 tokens)
"Could you please help me to write a summary of the following text?"

# 精簡版本 (更少 tokens)
"Summarize this text:"
```

### 2. 使用縮寫和符號
```python
# 範例
"and" → "&"
"number" → "#"
```

### 3. 移除不必要的空白和格式
```python
# 冗長
text = """
    This is a text
    with lots of
    extra spaces
"""

# 精簡
text = "This is a text with lots of extra spaces"
```

### 4. 對於長文本使用摘要
先摘要再處理，而不是直接送入完整文本。

## Token 與模型性能

### Token 數量與品質的關係
- **太少**：資訊不足，回應品質差
- **太多**：成本高，可能超出限制，且模型可能無法有效利用所有資訊
- **適中**：平衡資訊量與成本

### Token 在訓練中的角色
- 訓練資料量通常以 Token 數計算（如 "訓練了 1 兆個 tokens"）
- 更多 Token 通常意味著更好的模型（在其他條件相同的情況下）

## 未來發展

1. **更高效的 Tokenization**：減少不同語言間的效率差異
2. **動態詞彙表**：能夠學習和加入新詞
3. **多模態 Token**：統一處理文本、圖像、音訊等
4. **更大的 Context Window**：處理更長的文本

