# LLM 學習筆記目錄

這是一份完整的大型語言模型（LLM）學習筆記，涵蓋從基礎概念到進階應用的各個主題。

## 📚 目錄

### 基礎概念

#### [01. Chat Model 與 Base Model](01_chat_model_vs_base_model.md)
- Base Model（基礎模型）的定義與特性
- Chat Model（對話模型）的定義與特性
- 兩者的主要差異與使用時機

#### [02. 常見的大型語言模型](02_popular_models.md)
- **ChatGPT**：OpenAI 的對話式 AI
- **Gemini**：Google 的多模態模型
- **BERT**：雙向編碼器模型
- **T5**：文本到文本轉換模型
- **GPT**：生成式預訓練 Transformer
- **LLaMA**：Meta 的開源模型
- **Gemma**：Google 的輕量級模型

### 訓練方法

#### [03. 模型訓練方法](03_training_methods.md)
- **Fine-tune（微調）**：
  - Full Fine-tuning
  - Parameter-Efficient Fine-tuning（LoRA、Adapter）
  - Instruction Fine-tuning
- **Instruct Model（指令模型）**
- **RLHF（Reinforcement Learning from Human Feedback）**：
  - 三階段訓練流程
  - Reward Model
  - PPO 最佳化

### 開發框架

#### [04. LLM Application Framework](04_llm_frameworks.md)
- **LangChain**：組件化的 LLM 應用框架
- **LangGraph**：有狀態的圖結構框架
- **LlamaIndex**：資料索引和檢索專家
- **AutoGen**：Multi-agent 對話框架
- **CrewAI**：基於角色的協作框架
- 框架選擇指南

### 文本處理

#### [05. Token 與 Tokenization](05_token_and_tokenization.md)
- Token 的定義與重要性
- 分詞方法：
  - Word-level Tokenization
  - Character-level Tokenization
  - Subword Tokenization（BPE、WordPiece、SentencePiece）
- Token 計數與成本管理
- Context Window 限制

#### [06. 詞嵌入技術](06_word_embeddings.md)
- **One-hot Encoding**：
  - 獨熱編碼的原理
  - 優缺點分析
- **Word2Vec**：
  - CBOW 和 Skip-gram 架構
  - 向量運算與類比推理
  - 實作範例
- 其他詞嵌入技術：GloVe、FastText
- 上下文嵌入：ELMo、BERT

### AI Agent 能力

#### [07. AI Agent 能力](07_agent_capabilities.md)
- **Reflection（反思）**：
  - Self-Refine
  - Reflexion
  - Chain of Verification
- **Tool Use（工具使用）**：
  - Function Calling
  - 常見工具類型
- **Planning（規劃）**：
  - Forward / Backward / Hierarchical Planning
  - Adaptive Planning
- **Multi-agent（多代理系統）**：
  - 協作、競爭、層級模式
  - Agent 通訊機制

### 提示詞工程

#### [08. 提示詞工程技術](08_prompting_techniques.md)
- **CoT（Chain of Thought，思維鏈）**：
  - Zero-shot CoT
  - Few-shot CoT
  - Self-Consistency CoT
  - Tree of Thoughts
- **ReAct（Reasoning + Acting）**：
  - Thought → Action → Observation 循環
  - 實作範例
  - 與 CoT 的比較

### 檢索增強生成

#### [09. RAG（Retrieval-Augmented Generation）](09_rag.md)
- RAG 的基本架構
- 詳細流程：
  - 知識庫建立（文件切分、向量化、儲存）
  - 查詢處理（檢索、增強、生成）
- 向量資料庫：FAISS、ChromaDB、Pinecone、Weaviate、Qdrant
- 進階技術：
  - Hybrid Search
  - Re-ranking
  - Query Expansion
  - HyDE
  - Multi-hop Reasoning
- 評估指標與挑戰

### 應用介面

#### [10. LLM Client 網站與介面](10_llm_clients.md)
- **Web-based Clients**：
  - ChatGPT、Claude、Gemini、Microsoft Copilot、Perplexity
- **Playground**：OpenAI Playground、Anthropic Console
- **本地 Client**：LM Studio、Ollama、GPT4All
- **IDE 整合**：GitHub Copilot、Cursor、Codeium
- **API Clients**：Python、REST API
- 選擇指南與實務建議

## 🎯 學習路徑建議

### 初學者路徑
1. 從 [02. 常見的大型語言模型](02_popular_models.md) 開始，瞭解各種模型
2. 學習 [01. Chat Model 與 Base Model](01_chat_model_vs_base_model.md)，理解基本概念
3. 實作 [10. LLM Client](10_llm_clients.md)，開始使用 LLM
4. 學習 [08. 提示詞工程技術](08_prompting_techniques.md)，提升使用效果

### 應用開發路徑
1. 瞭解 [04. LLM Framework](04_llm_frameworks.md)，選擇合適的開發框架
2. 學習 [09. RAG](09_rag.md)，建構知識庫應用
3. 研究 [07. AI Agent 能力](07_agent_capabilities.md)，開發智能代理
4. 深入 [05. Token 與 Tokenization](05_token_and_tokenization.md)，最佳化成本

### 研究進階路徑
1. 深入 [03. 模型訓練方法](03_training_methods.md)，瞭解訓練流程
2. 研究 [06. 詞嵌入技術](06_word_embeddings.md)，理解底層原理
3. 探索 [07. AI Agent 能力](07_agent_capabilities.md)，研究前沿技術
4. 實驗 [08. 提示詞工程技術](08_prompting_techniques.md)，開發新技術

## 📖 專有名詞索引

### A-C
- **Agent**：智能代理 → [07. AI Agent 能力](07_agent_capabilities.md)
- **API**：應用程式介面 → [10. LLM Client](10_llm_clients.md)
- **AutoGen**：Multi-agent 框架 → [04. LLM Framework](04_llm_frameworks.md)
- **Base Model**：基礎模型 → [01. Chat Model vs Base Model](01_chat_model_vs_base_model.md)
- **BERT**：雙向編碼器模型 → [02. 常見模型](02_popular_models.md)
- **BPE**：位元組對編碼 → [05. Token](05_token_and_tokenization.md)
- **CBOW**：連續詞袋模型 → [06. 詞嵌入](06_word_embeddings.md)
- **Chat Model**：對話模型 → [01. Chat Model vs Base Model](01_chat_model_vs_base_model.md)
- **ChatGPT**：OpenAI 對話模型 → [02. 常見模型](02_popular_models.md)
- **ChromaDB**：向量資料庫 → [09. RAG](09_rag.md)
- **Claude**：Anthropic 模型 → [02. 常見模型](02_popular_models.md)
- **CoT**：思維鏈 → [08. 提示詞工程](08_prompting_techniques.md)
- **Context Window**：上下文視窗 → [05. Token](05_token_and_tokenization.md)
- **CrewAI**：角色協作框架 → [04. LLM Framework](04_llm_frameworks.md)

### D-L
- **Embedding**：嵌入向量 → [06. 詞嵌入](06_word_embeddings.md)
- **FAISS**：向量搜尋引擎 → [09. RAG](09_rag.md)
- **FastText**：子詞嵌入 → [06. 詞嵌入](06_word_embeddings.md)
- **Fine-tune**：微調 → [03. 訓練方法](03_training_methods.md)
- **Function Calling**：函式呼叫 → [07. Agent 能力](07_agent_capabilities.md)
- **Gemini**：Google 模型 → [02. 常見模型](02_popular_models.md)
- **Gemma**：Google 輕量模型 → [02. 常見模型](02_popular_models.md)
- **GloVe**：全域向量 → [06. 詞嵌入](06_word_embeddings.md)
- **GPT**：生成式預訓練 Transformer → [02. 常見模型](02_popular_models.md)
- **HyDE**：假設文件嵌入 → [09. RAG](09_rag.md)
- **Instruct Model**：指令模型 → [03. 訓練方法](03_training_methods.md)
- **LangChain**：LLM 應用框架 → [04. LLM Framework](04_llm_frameworks.md)
- **LangGraph**：圖結構框架 → [04. LLM Framework](04_llm_frameworks.md)
- **LLaMA**：Meta 開源模型 → [02. 常見模型](02_popular_models.md)
- **LlamaIndex**：資料索引框架 → [04. LLM Framework](04_llm_frameworks.md)
- **LoRA**：低秩適應 → [03. 訓練方法](03_training_methods.md)

### M-Z
- **Multi-agent**：多代理系統 → [07. Agent 能力](07_agent_capabilities.md)
- **One-hot**：獨熱編碼 → [06. 詞嵌入](06_word_embeddings.md)
- **Pinecone**：向量資料庫 → [09. RAG](09_rag.md)
- **Planning**：規劃 → [07. Agent 能力](07_agent_capabilities.md)
- **Playground**：測試介面 → [10. LLM Client](10_llm_clients.md)
- **Prompt Engineering**：提示詞工程 → [08. 提示詞工程](08_prompting_techniques.md)
- **RAG**：檢索增強生成 → [09. RAG](09_rag.md)
- **ReAct**：推理與行動 → [08. 提示詞工程](08_prompting_techniques.md)
- **Reflection**：反思 → [07. Agent 能力](07_agent_capabilities.md)
- **Reflexion**：反思機制 → [07. Agent 能力](07_agent_capabilities.md)
- **Re-ranking**：重新排序 → [09. RAG](09_rag.md)
- **Reward Model**：獎勵模型 → [03. 訓練方法](03_training_methods.md)
- **RLHF**：人類反饋強化學習 → [03. 訓練方法](03_training_methods.md)
- **SentencePiece**：分詞工具 → [05. Token](05_token_and_tokenization.md)
- **Skip-gram**：跳字模型 → [06. 詞嵌入](06_word_embeddings.md)
- **T5**：文本到文本 Transformer → [02. 常見模型](02_popular_models.md)
- **Token**：文本單元 → [05. Token](05_token_and_tokenization.md)
- **Tokenization**：分詞 → [05. Token](05_token_and_tokenization.md)
- **Tool Use**：工具使用 → [07. Agent 能力](07_agent_capabilities.md)
- **Vector Database**：向量資料庫 → [09. RAG](09_rag.md)
- **Weaviate**：向量搜尋引擎 → [09. RAG](09_rag.md)
- **Word2Vec**：詞向量 → [06. 詞嵌入](06_word_embeddings.md)
- **WordPiece**：詞片段編碼 → [05. Token](05_token_and_tokenization.md)

## 🔍 搜尋關鍵字

如果你想查詢特定主題，可以使用以下關鍵字：

- **模型選擇**：查看 [02. 常見模型](02_popular_models.md)
- **成本最佳化**：查看 [05. Token](05_token_and_tokenization.md) 和 [10. Client](10_llm_clients.md)
- **應用開發**：查看 [04. Framework](04_llm_frameworks.md)
- **知識庫建構**：查看 [09. RAG](09_rag.md)
- **提升準確度**：查看 [08. Prompting](08_prompting_techniques.md)
- **智能代理**：查看 [07. Agent](07_agent_capabilities.md)
- **模型訓練**：查看 [03. 訓練方法](03_training_methods.md)

## 💡 實用資源

### 官方文件
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Anthropic Documentation](https://docs.anthropic.com)
- [Google AI Documentation](https://ai.google.dev)
- [LangChain Documentation](https://python.langchain.com)
- [LlamaIndex Documentation](https://docs.llamaindex.ai)

### 學習資源
- [Hugging Face Course](https://huggingface.co/learn)
- [DeepLearning.AI Courses](https://www.deeplearning.ai)
- [Prompt Engineering Guide](https://www.promptingguide.ai)

### 社群資源
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
- [Hugging Face Community](https://huggingface.co/community)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

---

**最後更新**：2025 年 11 月

**作者**：根據網路資料彙整與個人學習整理

**授權**：僅供學習參考使用
