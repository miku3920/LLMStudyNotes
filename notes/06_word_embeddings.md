# 詞嵌入技術（Word Embeddings）

## 什麼是詞嵌入？

詞嵌入（Word Embedding）是將詞彙轉換為固定維度的連續向量表示的技術，使得語義相似的詞在向量空間中距離較近。

### 為什麼需要詞嵌入？
1. **機器學習需要數值輸入**：模型無法直接處理文字，需要將文字轉換為數字
2. **保留語義資訊**：相似的詞應該有相似的表示
3. **降維**：相比 One-hot，詞嵌入維度更低，更高效

## One-hot Encoding（獨熱編碼）

### 定義
One-hot Encoding 是最簡單的詞表示方法，將每個詞表示為一個高維稀疏向量，其中只有一個位置為 1，其餘為 0。

### 範例

假設詞彙表：`["貓", "狗", "鳥", "魚"]`

```
貓 → [1, 0, 0, 0]
狗 → [0, 1, 0, 0]
鳥 → [0, 0, 1, 0]
魚 → [0, 0, 0, 1]
```

### Python 實作

```python
import numpy as np

# 詞彙表
vocab = ["貓", "狗", "鳥", "魚"]
vocab_size = len(vocab)

# 建立詞到索引的映射
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

def one_hot_encode(word):
    """將詞轉換為 one-hot 向量"""
    vector = np.zeros(vocab_size)
    vector[word_to_idx[word]] = 1
    return vector

# 測試
print(one_hot_encode("貓"))  # [1. 0. 0. 0.]
print(one_hot_encode("狗"))  # [0. 1. 0. 0.]
```

### 使用 sklearn

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

words = np.array([["貓"], ["狗"], ["鳥"], ["魚"]])
encoder = OneHotEncoder(sparse=False)
encoder.fit(words)

print(encoder.transform([["貓"]]))  # [[1. 0. 0. 0.]]
```

### 優點
- **簡單直觀**：容易理解和實作
- **沒有假設**：不對詞之間的關係做任何假設

### 缺點
1. **維度災難**：詞彙表有 10 萬個詞，就需要 10 萬維的向量
2. **稀疏性**：絕大部分元素都是 0，浪費記憶體和運算
3. **無語義資訊**：所有詞之間的距離都相等，無法表達相似性
   - "貓" 和 "狗" 的距離 = "貓" 和 "汽車" 的距離
4. **OOV 問題**：無法處理未見過的詞

### 適用場景
- 詞彙表很小的情況
- 類別型變數的編碼（如標籤分類）
- 作為基準比較

## Word2Vec

### 定義
Word2Vec 是 Google 在 2013 年提出的詞嵌入技術，透過神經網路學習詞的分散式表示（Distributed Representation），使語義相似的詞在向量空間中接近。

### 核心思想：分佈假說（Distributional Hypothesis）
> "You shall know a word by the company it keeps."
> （從一個詞的鄰居詞可以瞭解這個詞的意思）

出現在相似上下文中的詞，語義也相似。

### 兩種架構

#### 1. CBOW (Continuous Bag of Words)

**目標**：根據上下文詞預測中心詞。

```
上下文: ["我", "喜歡", "___", "和", "狗"]
預測: "貓"
```

**特點**：
- 速度較快
- 適合大型資料集
- 對高頻詞效果較好

#### 2. Skip-gram

**目標**：根據中心詞預測上下文詞。

```
中心詞: "貓"
預測上下文: ["我", "喜歡", "和", "狗"]
```

**特點**：
- 對低頻詞效果較好
- 效果通常優於 CBOW
- 訓練較慢

### 架構圖示

```
CBOW:
[上下文1] ─┐
[上下文2] ─┤
[上下文3] ─┼─→ [隱藏層] ─→ [中心詞]
[上下文4] ─┘

Skip-gram:
              ┌─→ [上下文1]
              ├─→ [上下文2]
[中心詞] ─→ [隱藏層] ┼─→ [上下文3]
              └─→ [上下文4]
```

### Word2Vec 的特性

#### 1. 語義相似性
相似的詞在向量空間中接近：

```python
similar_to_king = ["queen", "prince", "monarch", "crown"]
similar_to_paris = ["london", "berlin", "france", "capital"]
```

#### 2. 向量運算（類比推理）

著名的範例：
```
king - man + woman ≈ queen
paris - france + italy ≈ rome
bigger - big + small ≈ smaller
```

### Python 實作（使用 Gensim）

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 準備訓練資料
sentences = [
    "我 喜歡 貓 和 狗",
    "貓 很 可愛",
    "狗 很 忠誠",
    "我 養 了 一 隻 貓",
    "狗 喜歡 玩 球"
]

# 將句子切分成詞列表
tokenized_sentences = [sentence.split() for sentence in sentences]

# 訓練 Word2Vec 模型
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # 向量維度
    window=5,             # 上下文視窗大小
    min_count=1,          # 最小詞頻
    workers=4,            # 平行處理數
    sg=0                  # 0=CBOW, 1=Skip-gram
)

# 獲取詞向量
cat_vector = model.wv['貓']
print(f"'貓' 的向量維度: {len(cat_vector)}")

# 找相似詞
similar_words = model.wv.most_similar('貓', topn=3)
print(f"與 '貓' 相似的詞: {similar_words}")

# 詞彙類比
result = model.wv.most_similar(positive=['狗', '可愛'], negative=['貓'], topn=1)
print(f"狗 - 貓 + 可愛 ≈ {result}")
```

### 使用預訓練的 Word2Vec 模型

```python
import gensim.downloader as api

# 載入 Google News 預訓練模型（需要下載，約 1.6GB）
model = api.load("word2vec-google-news-300")

# 查詢相似詞
print(model.most_similar("king", topn=5))

# 詞彙類比
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)  # [('queen', 0.7118...)]

# 計算相似度
similarity = model.similarity('cat', 'dog')
print(f"cat 和 dog 的相似度: {similarity}")
```

### 超參數調整

#### vector_size（向量維度）
- **小**（50-100）：訓練快，但表達能力有限
- **中**（200-300）：常用的選擇，平衡效能與效果
- **大**（500+）：表達能力強，但需要更多資料和運算

#### window（視窗大小）
- **小**（2-3）：捕捉語法（syntactic）關係
- **大**（5-10）：捕捉語義（semantic）關係

#### min_count（最小詞頻）
- 過濾低頻詞，減少噪音和訓練時間
- 通常設為 5 或 10

### Word2Vec 的訓練技巧

#### 1. Negative Sampling
不計算完整的 softmax（太慢），而是隨機採樣負例（非上下文詞）。

#### 2. Hierarchical Softmax
使用 Huffman 樹結構來加速訓練。

#### 3. Subsampling（下採樣）
降低高頻詞（如 "the"、"a"）的採樣機率，避免它們主導訓練。

## One-hot vs Word2Vec 比較

| 特性 | One-hot | Word2Vec |
|------|---------|----------|
| 維度 | 詞彙表大小（10k-100k+） | 固定（50-300） |
| 稀疏性 | 稀疏（99.99%是0） | 稠密（所有元素有值） |
| 語義 | 無語義資訊 | 保留語義相似性 |
| 計算 | 簡單 | 需要訓練 |
| 記憶體 | 大 | 小 |
| OOV | 無法處理 | 無法處理（但可用子詞） |
| 適用 | 小詞彙表、類別變數 | 大詞彙表、NLP 任務 |

## 其他詞嵌入技術

### GloVe (Global Vectors)
- Stanford 開發
- 結合全域統計資訊和局部上下文
- 與 Word2Vec 性能相當

### FastText
- Facebook 開發
- 考慮子詞資訊（subword）
- 可以處理 OOV（未知詞）
- 對拼字錯誤和罕見詞效果好

```python
from gensim.models import FastText

model = FastText(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=1
)

# 即使 "貓咪" 沒在訓練資料中，也能生成向量
vector = model.wv['貓咪']
```

### Contextualized Embeddings（上下文嵌入）
- **ELMo**：雙向 LSTM
- **BERT**：雙向 Transformer
- **GPT**：單向 Transformer

**關鍵差異**：同一個詞在不同上下文中有不同的向量

```
"I went to the bank to deposit money."  # bank = 銀行
"I sat on the bank of the river."       # bank = 河岸
```

Word2Vec：兩個 "bank" 有相同的向量
BERT：兩個 "bank" 有不同的向量（根據上下文）

## 詞嵌入的應用

### 1. 文本分類
將詞向量平均或加總作為文本表示。

```python
import numpy as np

def sentence_to_vector(sentence, model):
    """將句子轉換為向量（詞向量平均）"""
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
```

### 2. 語義搜尋
找到與查詢語義相似的文件。

### 3. 推薦系統
將商品、使用者描述轉換為向量，計算相似度。

### 4. 機器翻譯
作為編碼器-解碼器模型的輸入層。

### 5. 情感分析
詞向量作為特徵輸入分類模型。

## 詞嵌入的限制

1. **一詞多義問題**：Word2Vec 無法區分同一個詞的不同意義（已被 BERT 等解決）
2. **需要大量資料**：高品質的詞嵌入需要大量訓練資料
3. **OOV 問題**：Word2Vec 無法處理訓練時未見過的詞（FastText 可以）
4. **語言依賴**：不同語言需要分別訓練
5. **偏見問題**：訓練資料中的偏見會反映在詞向量中

## 偏見範例

```python
# 性別偏見
model.most_similar(positive=['doctor', 'woman'], negative=['man'])
# 可能傾向 'nurse'

model.most_similar(positive=['nurse', 'man'], negative=['woman'])
# 可能傾向 'doctor'
```

## 現代 NLP 的演進

```
One-hot (1980s)
    ↓
Word2Vec / GloVe (2013-2014)
    ↓
ELMo (2018) - 上下文嵌入
    ↓
BERT / GPT (2018-2019) - Transformer
    ↓
GPT-3 / GPT-4 (2020-2023) - 大型語言模型
```

## 總結與建議

### 選擇建議

**使用 One-hot**：
- 詞彙表非常小（< 100 詞）
- 類別變數編碼
- 不需要語義資訊

**使用 Word2Vec / FastText**：
- 需要靜態詞向量
- 計算資源有限
- 需要可解釋的詞嵌入

**使用 BERT / GPT**：
- 需要上下文相關的表示
- 追求最佳效能
- 有足夠計算資源

### 實務應用
- 大部分現代 NLP 任務已經使用 Transformer 模型
- Word2Vec 仍然在某些場景有用（如輕量級應用、詞彙分析）
- 理解詞嵌入的原理有助於理解現代 LLM 的工作方式

