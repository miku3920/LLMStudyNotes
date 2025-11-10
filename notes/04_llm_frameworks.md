# LLM Application Framework（LLM 應用框架）

## 什麼是 LLM Framework？

LLM Framework 是用於簡化大型語言模型應用開發的工具和函式庫，提供預建的元件和抽象層，讓開發者能更快速地建構基於 LLM 的應用。

## LangChain

### 定義
LangChain 是最早也最流行的 LLM 應用開發框架，提供組件化的工具來組合模型、記憶體、資料來源和其他工具。

### 核心概念

#### 1. Chains（鏈）
將多個元件串接起來，形成完整的工作流程。

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# 建立提示詞範本
prompt = PromptTemplate(
    input_variables=["product"],
    template="為 {product} 寫一個吸引人的廣告標語"
)

# 建立鏈
chain = LLMChain(llm=OpenAI(), prompt=prompt)

# 執行
result = chain.run("環保水壺")
```

#### 2. Agents（代理）
能夠根據使用者輸入，自主決定要使用哪些工具，並執行多步驟推理。

#### 3. Memory（記憶體）
保存對話歷史，使模型能夠參考先前的互動。

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "你好"}, {"output": "您好！有什麼可以幫助您的？"})
```

#### 4. Tools（工具）
讓 LLM 能夠呼叫外部功能，如搜尋、計算、查詢資料庫等。

```python
from langchain.tools import Tool

def calculator(expression):
    return eval(expression)

calc_tool = Tool(
    name="Calculator",
    func=calculator,
    description="用於數學計算"
)
```

#### 5. Retrievers（檢索器）
用於從外部知識庫檢索相關資訊。

### 優點
- 生態系統成熟，社群活躍
- 文件豐富，範例多
- 支援多種 LLM 和向量資料庫
- 元件豐富且可組合

### 缺點
- API 變動頻繁
- 對於複雜應用，抽象層可能過度
- 效能開銷較大

### 適用場景
- 快速原型開發
- 需要整合多種工具和資料來源
- 標準的 RAG 應用

## LangGraph

### 定義
LangGraph 是 LangChain 團隊開發的進階框架，專門用於建構有狀態的、循環的 LLM 應用。

### 核心特點
- **圖結構**：使用有向圖（Directed Graph）定義工作流程
- **狀態管理**：明確的狀態管理機制
- **循環支援**：支援迴圈和條件分支
- **人機協作**：可以在流程中插入人工審核

### 與 LangChain 的差異
- LangChain：線性的鏈式結構
- LangGraph：靈活的圖結構，支援複雜的控制流

### 範例場景
```python
from langgraph.graph import StateGraph

# 定義狀態
class State:
    messages: list
    next_step: str

# 建立圖
workflow = StateGraph(State)

# 添加節點
workflow.add_node("researcher", research_node)
workflow.add_node("writer", write_node)
workflow.add_node("reviewer", review_node)

# 定義邊（流程）
workflow.add_edge("researcher", "writer")
workflow.add_conditional_edges(
    "reviewer",
    should_continue,  # 決策函式
    {
        "continue": "writer",
        "end": END
    }
)
```

### 適用場景
- 複雜的多步驟工作流程
- 需要人工審核的應用
- Multi-agent 系統
- 需要循環和條件分支的邏輯

## LlamaIndex

### 定義
LlamaIndex（原名 GPT Index）專注於將 LLM 與私有資料連接，特別擅長資料索引和檢索。

### 核心功能

#### 1. Data Connectors（資料連接器）
從各種來源載入資料：
- 文件（PDF、Word、Markdown）
- 網頁
- 資料庫
- API

#### 2. Indexing（索引）
將資料建立成可高效查詢的索引結構：
- **Vector Store Index**：向量檢索
- **Tree Index**：樹狀結構
- **Keyword Index**：關鍵字索引
- **Knowledge Graph Index**：知識圖譜

#### 3. Query Engine（查詢引擎）
提供自然語言查詢介面。

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# 載入文件
documents = SimpleDirectoryReader('data').load_data()

# 建立索引
index = VectorStoreIndex.from_documents(documents)

# 查詢
query_engine = index.as_query_engine()
response = query_engine.query("公司的營收成長率是多少？")
print(response)
```

### 優點
- 專注於資料索引和檢索，在這方面做得非常好
- 支援多種索引策略
- 文件和範例清晰
- 與 LangChain 可以互相整合

### 適用場景
- RAG（Retrieval-Augmented Generation）應用
- 企業知識庫查詢
- 文件問答系統
- 需要處理大量私有資料

## AutoGen

### 定義
AutoGen 是微軟開發的 Multi-agent 對話框架，支援多個 Agent 之間的協作對話。

### 核心概念

#### 1. Conversable Agents（可對話代理）
可以互相對話的智能體。

```python
from autogen import AssistantAgent, UserProxyAgent

# 助理代理
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"}
)

# 使用者代理
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"}
)

# 啟動對話
user_proxy.initiate_chat(
    assistant,
    message="寫一個 Python 程式計算費氏數列"
)
```

#### 2. Code Execution（程式碼執行）
Agent 可以生成並執行程式碼，實現自動化任務。

#### 3. Multi-agent Collaboration（多代理協作）
定義多個具有不同角色的 Agent，讓它們協作完成複雜任務。

### 特點
- 自動化的 Agent 對話
- 內建程式碼生成和執行能力
- 支援人工介入和反饋
- 適合自動化工作流程

### 適用場景
- 程式碼生成和除錯
- 自動化資料分析
- Multi-agent 系統
- 需要迭代式改進的任務

## CrewAI

### 定義
CrewAI 是專為生產環境設計的 Multi-agent 協作框架，強調角色分工和任務編排。

### 核心概念

#### 1. Roles（角色）
每個 Agent 有明確的角色定義。

```python
from crewai import Agent, Task, Crew

# 定義研究員
researcher = Agent(
    role='市場研究員',
    goal='收集並分析市場趨勢',
    backstory='你是一位經驗豐富的市場分析師',
    verbose=True
)

# 定義寫手
writer = Agent(
    role='內容寫手',
    goal='撰寫吸引人的市場報告',
    backstory='你是一位專業的商業寫手',
    verbose=True
)
```

#### 2. Tasks（任務）
定義具體的任務和執行順序。

```python
task1 = Task(
    description='研究 AI 產業的最新趨勢',
    agent=researcher
)

task2 = Task(
    description='根據研究結果撰寫報告',
    agent=writer
)
```

#### 3. Crew（團隊）
組織 Agent 和任務形成團隊。

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2
)

result = crew.kickoff()
```

### 執行模式
- **Sequential（順序）**：任務依序執行
- **Hierarchical（層級）**：有管理者協調任務
- **Consensus（共識）**：多個 Agent 達成共識

### 優點
- 概念清晰，易於理解
- 適合模擬真實團隊協作
- 生產環境友善
- 任務編排直觀

### 適用場景
- 需要多個專業角色協作的任務
- 內容創作工作流程
- 市場研究和報告生成
- 企業自動化流程

## Framework 選擇指南

### LangChain
**使用時機**：
- 快速原型開發
- 標準的 RAG 應用
- 需要豐富的生態系統和預建元件

**範例專案**：
- 文件問答聊天機器人
- 客服自動化
- 簡單的 Agent 應用

### LangGraph
**使用時機**：
- 複雜的狀態管理
- 需要循環和條件分支
- 人機協作工作流程

**範例專案**：
- 多輪對話系統
- 需要人工審核的內容生成
- 複雜的決策流程

### LlamaIndex
**使用時機**：
- 主要需求是資料索引和檢索
- 處理大量文件
- 企業知識管理

**範例專案**：
- 企業內部知識庫
- 法律文件查詢
- 學術論文問答

### AutoGen
**使用時機**：
- 需要程式碼生成和執行
- 自動化開發流程
- 迭代式問題解決

**範例專案**：
- 自動化程式開發助手
- 資料分析自動化
- 程式碼審查和除錯

### CrewAI
**使用時機**：
- 需要模擬團隊協作
- 角色分工明確的任務
- 生產環境應用

**範例專案**：
- 市場研究報告生成
- 內容創作流水線
- 複雜的商業流程自動化

## 組合使用

這些框架並非互斥，可以根據需求組合使用：

- **LangChain + LlamaIndex**：用 LlamaIndex 處理資料索引，LangChain 處理對話流程
- **LangGraph + AutoGen**：用 LangGraph 管理整體流程，AutoGen 處理特定的 Agent 協作
- **CrewAI + LlamaIndex**：CrewAI 組織團隊協作，LlamaIndex 提供知識檢索

## 效能與成本考量

- **LangChain**：較高的抽象成本，適合開發階段
- **LangGraph**：更精細的控制，可最佳化效能
- **LlamaIndex**：專注於檢索，效能優異
- **AutoGen / CrewAI**：Multi-agent 系統通常成本較高（多次 LLM 呼叫）

## 未來趨勢

1. **標準化**：各框架逐漸趨向互通和標準化
2. **效能最佳化**：減少不必要的 LLM 呼叫
3. **可觀測性**：更好的除錯和監控工具
4. **生產就緒**：從原型到生產環境的完整支援

