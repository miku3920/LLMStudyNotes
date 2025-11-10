# RAG（Retrieval-Augmented Generation）

## 定義

RAG（Retrieval-Augmented Generation，檢索增強生成）是一種結合資訊檢索和文本生成的技術，讓 LLM 在生成回應時能夠參考外部知識庫，提升回應的準確性和可信度。

## 為什麼需要 RAG？

### LLM 的限制

1. **知識截止日期**：訓練資料有時間限制，無法獲取最新資訊
2. **幻覺問題（Hallucination）**：可能生成聽起來合理但實際錯誤的內容
3. **領域知識不足**：對特定領域或企業內部資料不瞭解
4. **無法引用來源**：難以追溯資訊來源

### RAG 的解決方案

- **動態知識**：可以隨時更新知識庫
- **準確性提升**：基於實際文件生成回應
- **可追溯性**：可以引用來源文件
- **成本效益**：不需要重新訓練模型

## RAG 的基本架構

```
使用者查詢
    ↓
1. 檢索 (Retrieval)
    - 將查詢轉換為向量
    - 在向量資料庫中搜尋相關文件
    ↓
2. 增強 (Augmentation)
    - 將查詢和檢索到的文件組合
    - 建立完整的 Prompt
    ↓
3. 生成 (Generation)
    - LLM 基於查詢和文件生成回應
    ↓
回應
```

## RAG 的詳細流程

### 階段一：知識庫建立（離線處理）

#### 1. 文件收集
收集相關的文件資料：
- PDF、Word、Markdown 等文件
- 網頁內容
- 資料庫記錄
- API 資料

#### 2. 文件切分（Chunking）
將長文件切分成較小的片段。

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """將文本切分成較小的片段"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重疊部分
    
    return chunks
```

**切分策略**：
- **固定長度**：每 N 個字元或 Token
- **句子邊界**：按句子切分
- **段落邊界**：按段落切分
- **語義切分**：按主題切分

#### 3. 向量化（Embedding）
將文件片段轉換為向量。

```python
from sentence_transformers import SentenceTransformer

# 使用預訓練的 Embedding 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = ["這是第一個文件片段", "這是第二個文件片段"]
embeddings = model.encode(chunks)
```

#### 4. 儲存到向量資料庫
將向量和原始文本儲存起來。

```python
import faiss
import numpy as np

# 建立 FAISS 索引
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 添加向量
index.add(embeddings.astype('float32'))
```

### 階段二：查詢處理（線上處理）

#### 1. 查詢向量化
將使用者查詢轉換為向量。

```python
query = "RAG 是什麼？"
query_embedding = model.encode([query])
```

#### 2. 相似度搜尋
在向量資料庫中搜尋最相關的文件。

```python
k = 5  # 檢索前 5 個最相關的文件
distances, indices = index.search(query_embedding.astype('float32'), k)

# 獲取相關文件
relevant_chunks = [chunks[i] for i in indices[0]]
```

#### 3. 建立增強 Prompt
將查詢和檢索到的文件組合。

```python
def build_rag_prompt(query, relevant_docs):
    context = "\n\n".join(relevant_docs)
    
    prompt = f"""
請根據以下參考資料回答問題。如果參考資料中沒有相關資訊，請明確說明。

參考資料：
{context}

問題：{query}

回答：
"""
    return prompt
```

#### 4. LLM 生成回應
使用 LLM 生成最終回應。

```python
prompt = build_rag_prompt(query, relevant_chunks)
response = llm.generate(prompt)
```

## 完整的 RAG 實作範例

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List

class SimpleRAG:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        """初始化 RAG 系統"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunks = []
        self.index = None
    
    def add_documents(self, documents: List[str], chunk_size=500):
        """添加文件到知識庫"""
        # 1. 切分文件
        for doc in documents:
            chunks = self.chunk_text(doc, chunk_size)
            self.chunks.extend(chunks)
        
        # 2. 向量化
        embeddings = self.embedding_model.encode(self.chunks)
        
        # 3. 建立索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"已添加 {len(self.chunks)} 個文件片段到知識庫")
    
    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """切分文本"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def retrieve(self, query: str, top_k=3) -> List[str]:
        """檢索相關文件"""
        # 查詢向量化
        query_embedding = self.embedding_model.encode([query])
        
        # 搜尋
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # 返回相關文件
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        return relevant_chunks
    
    def query(self, question: str, llm, top_k=3) -> str:
        """RAG 查詢"""
        # 1. 檢索相關文件
        relevant_docs = self.retrieve(question, top_k)
        
        # 2. 建立 Prompt
        context = "\n\n---\n\n".join(relevant_docs)
        prompt = f"""
請根據以下參考資料回答問題。請盡可能引用參考資料中的內容。
如果參考資料中沒有相關資訊，請說明「根據提供的資料無法回答此問題」。

參考資料：
{context}

問題：{question}

回答："""
        
        # 3. LLM 生成
        response = llm.generate(prompt)
        
        return response, relevant_docs

# 使用範例
rag = SimpleRAG()

# 添加文件
documents = [
    """RAG 是 Retrieval-Augmented Generation 的縮寫，
    是一種結合檢索和生成的技術。它讓 LLM 能夠參考外部知識庫，
    提升回應的準確性。""",
    
    """向量資料庫如 FAISS、Pinecone、Weaviate 等，
    是 RAG 系統的核心元件，用於高效地儲存和檢索向量。""",
]

rag.add_documents(documents)

# 查詢
response, sources = rag.query("什麼是 RAG？", llm)
print("回應：", response)
print("\n參考來源：")
for i, source in enumerate(sources, 1):
    print(f"{i}. {source[:100]}...")
```

## 向量資料庫

### 常見的向量資料庫

#### 1. FAISS（Facebook AI Similarity Search）
- Meta 開源
- 高效的相似度搜尋
- 適合本地部署

```python
import faiss

# 建立索引
index = faiss.IndexFlatL2(dimension)  # L2 距離
# 或
index = faiss.IndexFlatIP(dimension)  # 內積（餘弦相似度）

# 添加向量
index.add(vectors)

# 搜尋
D, I = index.search(query_vectors, k=5)
```

#### 2. ChromaDB
- 開源、易用
- 內建 Embedding 功能
- 支援本地和雲端

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_collection")

# 添加文件
collection.add(
    documents=["文件1", "文件2"],
    ids=["id1", "id2"]
)

# 查詢
results = collection.query(
    query_texts=["查詢文本"],
    n_results=5
)
```

#### 3. Pinecone
- 雲端向量資料庫
- 高性能、可擴展
- 商業產品

#### 4. Weaviate
- 開源向量搜尋引擎
- 支援混合搜尋（向量 + 關鍵字）
- 可自建或使用雲端服務

#### 5. Qdrant
- 開源向量搜尋引擎
- Rust 開發，高效能
- 支援篩選和混合搜尋

### 選擇向量資料庫的考量

| 資料庫 | 適用場景 | 優點 | 缺點 |
|--------|---------|------|------|
| FAISS | 本地、實驗 | 快速、免費 | 需自行管理 |
| ChromaDB | 小型專案 | 簡單易用 | 規模有限 |
| Pinecone | 生產環境 | 可擴展、管理簡單 | 需付費 |
| Weaviate | 企業應用 | 功能豐富 | 較複雜 |
| Qdrant | 高效能需求 | 快速、功能完整 | 生態較新 |

## RAG 的進階技術

### 1. Hybrid Search（混合搜尋）
結合向量搜尋和關鍵字搜尋。

```python
# 向量搜尋分數
vector_results = vector_search(query)

# 關鍵字搜尋分數（BM25）
keyword_results = keyword_search(query)

# 結合分數
final_results = combine_scores(vector_results, keyword_results, 
                                 alpha=0.7)  # 70% 向量，30% 關鍵字
```

### 2. Re-ranking（重新排序）
使用更精確的模型對檢索結果重新排序。

```python
from sentence_transformers import CrossEncoder

# 初步檢索（快速但較不精確）
candidates = vector_search(query, top_k=100)

# 重新排序（慢但更精確）
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
scores = reranker.predict([(query, doc) for doc in candidates])

# 選擇分數最高的
top_docs = [candidates[i] for i in np.argsort(scores)[-5:]]
```

### 3. Query Expansion（查詢擴展）
擴展原始查詢以提高檢索召回率。

```python
def expand_query(query, llm):
    """使用 LLM 生成相關的查詢變體"""
    prompt = f"""
原始查詢：{query}

請生成 3 個相關的查詢變體，幫助搜尋更多相關資訊：
1. 
2. 
3. 
"""
    expansions = llm.generate(prompt)
    return [query] + parse_expansions(expansions)

# 使用多個查詢檢索
expanded_queries = expand_query(original_query, llm)
all_results = []
for q in expanded_queries:
    results = retrieve(q)
    all_results.extend(results)

# 去重並排序
final_results = deduplicate_and_rank(all_results)
```

### 4. Hypothetical Document Embeddings (HyDE)
生成假設性的答案文件，用其向量來檢索。

```python
def hyde_retrieval(query, llm, rag_system):
    # 1. 生成假設性答案
    hypothetical_answer = llm.generate(
        f"請回答以下問題：{query}"
    )
    
    # 2. 使用假設性答案來檢索
    # (因為答案和相關文件更相似)
    results = rag_system.retrieve(hypothetical_answer)
    
    return results
```

### 5. Multi-hop Reasoning（多跳推理）
多次檢索來回答複雜問題。

```python
def multi_hop_rag(question, rag_system, llm, max_hops=3):
    """多跳 RAG"""
    context = []
    current_query = question
    
    for hop in range(max_hops):
        # 檢索
        docs = rag_system.retrieve(current_query)
        context.extend(docs)
        
        # 生成中間答案或下一個查詢
        prompt = f"""
根據以下資訊，回答問題或提出需要進一步查詢的子問題：

已知資訊：
{' '.join(context)}

問題：{question}

下一步：
"""
        next_step = llm.generate(prompt)
        
        # 判斷是否需要繼續
        if is_final_answer(next_step):
            return next_step
        
        current_query = extract_subquery(next_step)
    
    return "無法回答此問題"
```

## RAG 的評估指標

### 1. 檢索品質
- **Recall@K**：前 K 個結果中包含相關文件的比例
- **Precision@K**：前 K 個結果中相關文件的比例
- **MRR（Mean Reciprocal Rank）**：第一個相關結果的排名倒數

### 2. 生成品質
- **Faithfulness**：生成內容是否忠實於檢索文件
- **Answer Relevance**：答案是否回答了問題
- **Context Relevance**：檢索的文件是否相關

### 3. 端到端評估
- **準確率**：答案是否正確
- **完整性**：答案是否完整
- **引用品質**：是否正確引用來源

## RAG 的挑戰與解決方案

### 挑戰 1：檢索不相關的文件
**解決**：
- 改進 Embedding 模型
- 使用混合搜尋
- 添加重新排序步驟
- 查詢擴展

### 挑戰 2：文件切分不當
**解決**：
- 使用語義切分
- 添加重疊部分
- 保留文件結構資訊

### 挑戰 3：LLM 忽略檢索文件
**解決**：
- 改進 Prompt 設計
- 明確指示要引用文件
- 使用 Few-shot 範例

### 挑戰 4：回應時間長
**解決**：
- 使用更快的向量資料庫
- 減少檢索文件數量
- 使用較小的 LLM
- 實作快取機制

## RAG vs Fine-tuning

| 特性 | RAG | Fine-tuning |
|------|-----|-------------|
| 更新知識 | 容易（更新資料庫） | 需要重新訓練 |
| 成本 | 較低 | 較高 |
| 可追溯性 | 高（可引用來源） | 低 |
| 客製化程度 | 中等 | 高 |
| 推理速度 | 較慢（需檢索） | 較快 |
| 適用場景 | 知識問答、文件查詢 | 特定任務、風格調整 |

## RAG 的應用場景

1. **企業知識庫**：員工查詢內部文件
2. **客服系統**：基於產品文件回答問題
3. **法律助手**：查詢法律條文和案例
4. **醫療問答**：基於醫學文獻回答問題
5. **教育助手**：基於教材回答學生問題
6. **程式碼助手**：基於程式碼庫回答技術問題

## 實務建議

### 文件準備
1. **清理資料**：移除無關內容、格式問題
2. **結構化**：保留標題、章節等結構資訊
3. **元資料**：添加來源、日期等元資料

### 切分策略
1. **適當大小**：通常 200-500 Token
2. **語義完整**：避免句子被切斷
3. **重疊**：10-20% 的重疊避免資訊遺失

### Prompt 設計
1. **明確指示**：要求基於文件回答
2. **處理不相關**：指示如何處理無關文件
3. **引用來源**：要求標註資訊來源

### 效能最佳化
1. **快取**：快取常見查詢
2. **預處理**：提前處理文件
3. **批次處理**：批次查詢減少開銷

## 總結

RAG 是現代 LLM 應用的核心技術之一，它：
- **彌補 LLM 的知識限制**
- **提升回應的準確性和可信度**
- **相比 Fine-tuning 更靈活且成本更低**
- **是企業部署 LLM 的主要方案**

理解並掌握 RAG 技術對於開發實用的 LLM 應用至關重要。

