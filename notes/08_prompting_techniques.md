# 提示詞工程技術（Prompting Techniques）

## 什麼是提示詞工程？

Prompting（提示詞工程）是設計和最佳化輸入文本（Prompt）的技術，以引導 LLM 生成期望的輸出。好的提示詞可以顯著提升模型的表現。

## CoT（Chain of Thought，思維鏈）

### 定義
Chain of Thought 是一種提示詞技術，透過引導模型逐步展示推理過程，來提升複雜問題的解決能力。

### 核心概念
讓模型「思考」並展示中間步驟，而不是直接給出答案。

### 基本範例

#### 沒有 CoT
```
問題：Roger 有 5 顆網球。他又買了 2 罐網球，每罐有 3 顆。他現在有幾顆網球？

模型回應：11 顆
```
（可能是對的，但不知道怎麼算出來的）

#### 使用 CoT
```
問題：Roger 有 5 顆網球。他又買了 2 罐網球，每罐有 3 顆。他現在有幾顆網球？
請一步步思考。

模型回應：
讓我一步步來：
1. Roger 原本有 5 顆網球
2. 他買了 2 罐，每罐 3 顆
3. 新買的網球數量 = 2 × 3 = 6 顆
4. 總共 = 5 + 6 = 11 顆

答案：11 顆網球
```

### CoT 的變體

#### 1. Zero-shot CoT
不需要提供範例，只需在問題後加上「讓我們一步步思考」。

```python
def zero_shot_cot(question):
    prompt = f"{question}\n\n讓我們一步步思考："
    return llm.generate(prompt)
```

**範例**：
```
問題：如果一個數除以 7 餘 3，除以 5 餘 2，這個數最小是多少？
讓我們一步步思考：

1. 除以 7 餘 3 的數：3, 10, 17, 24, 31, 38, 45...
2. 除以 5 餘 2 的數：2, 7, 12, 17, 22, 27, 32, 37...
3. 同時滿足兩個條件的最小數是 17

答案：17
```

#### 2. Few-shot CoT
提供幾個帶有推理過程的範例，讓模型學習這種推理模式。

```python
def few_shot_cot(question):
    prompt = """
範例 1：
問題：Mary 有 3 顆蘋果，她給了 John 2 顆，她還剩幾顆？
思考過程：
1. Mary 原本有 3 顆蘋果
2. 她給了 John 2 顆
3. 剩下 = 3 - 2 = 1 顆
答案：1 顆

範例 2：
問題：一個數加上 5 等於 12，這個數是多少？
思考過程：
1. 設這個數為 x
2. x + 5 = 12
3. x = 12 - 5 = 7
答案：7

現在請解決這個問題：
問題：{question}
思考過程：
"""
    return llm.generate(prompt)
```

#### 3. Self-Consistency CoT
多次生成推理過程，選擇最常出現的答案（投票機制）。

```python
def self_consistency_cot(question, num_samples=5):
    """使用自洽性提升準確度"""
    answers = []
    
    for _ in range(num_samples):
        prompt = f"{question}\n讓我們一步步思考："
        response = llm.generate(prompt, temperature=0.7)
        answer = extract_final_answer(response)
        answers.append(answer)
    
    # 投票選出最常見的答案
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    
    return most_common
```

#### 4. Tree of Thoughts (ToT)
探索多個推理路徑，形成思考樹，選擇最佳路徑。

```
                問題
               /  |  \
          思路1 思路2 思路3
         /  \   |     /  \
      步驟  步驟 步驟  步驟 步驟
       ...
```

### CoT 的優勢
1. **提升準確度**：特別是在數學、邏輯推理任務
2. **可解釋性**：能看到模型的推理過程
3. **錯誤診斷**：可以找出推理中的錯誤環節
4. **複雜問題**：能處理多步驟推理

### CoT 的限制
1. **輸出更長**：需要更多 Token
2. **時間更久**：生成時間增加
3. **不適用所有任務**：簡單任務反而可能變差
4. **可能出現幻覺**：推理過程可能包含錯誤

### 實作範例

```python
class ChainOfThoughtAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def solve(self, problem, use_cot=True):
        if use_cot:
            prompt = f"""
問題：{problem}

請按照以下步驟解決：
1. 理解問題：重新表述問題
2. 分析：找出關鍵資訊
3. 推理：逐步推導
4. 驗證：檢查答案是否合理
5. 結論：給出最終答案

讓我們開始：
"""
        else:
            prompt = f"問題：{problem}\n答案："
        
        return self.llm.generate(prompt)
```

### 應用場景
- 數學問題
- 邏輯推理
- 因果分析
- 策略規劃
- 程式碼除錯

## ReAct（Reasoning + Acting）

### 定義
ReAct 是結合推理（Reasoning）和行動（Acting）的框架，讓模型在推理過程中能夠與環境互動，執行行動來獲取資訊。

### 核心概念
```
Thought（思考） → Action（行動） → Observation（觀察） → Thought → ...
```

模型不只是思考，還能採取行動（如搜尋、計算）來獲取必要的資訊。

### ReAct 的流程

1. **Thought**：模型思考下一步該做什麼
2. **Action**：執行某個行動（呼叫工具）
3. **Observation**：獲得行動的結果
4. **重複**：直到得到最終答案

### 實作範例

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools  # 可用的工具字典
    
    def run(self, question, max_steps=10):
        prompt_history = f"問題：{question}\n\n"
        
        for step in range(max_steps):
            # 生成 Thought 和 Action
            prompt = prompt_history + "Thought:"
            response = self.llm.generate(prompt)
            
            # 解析回應
            thought, action, action_input = self.parse_response(response)
            
            prompt_history += f"Thought {step+1}: {thought}\n"
            prompt_history += f"Action {step+1}: {action}[{action_input}]\n"
            
            # 檢查是否結束
            if action == "Finish":
                return action_input
            
            # 執行 Action
            observation = self.execute_action(action, action_input)
            prompt_history += f"Observation {step+1}: {observation}\n\n"
        
        return "達到最大步驟數"
    
    def execute_action(self, action, action_input):
        """執行指定的行動"""
        if action in self.tools:
            return self.tools[action](action_input)
        else:
            return f"錯誤：找不到工具 {action}"
    
    def parse_response(self, response):
        """解析模型回應，提取 Thought、Action 和 Action Input"""
        # 這裡需要根據實際格式解析
        # 簡化範例
        lines = response.strip().split('\n')
        thought = lines[0] if lines else ""
        action = "Search"  # 簡化
        action_input = "query"
        return thought, action, action_input
```

### ReAct 範例對話

```
問題：2024 年諾貝爾物理學獎得主是誰？

Thought 1: 我需要搜尋 2024 年諾貝爾物理學獎的資訊
Action 1: Search[2024 諾貝爾物理學獎]
Observation 1: 2024 年諾貝爾物理學獎授予 John Hopfield 和 Geoffrey Hinton，
               表彰他們在人工神經網路方面的開創性工作

Thought 2: 我已經找到答案了
Action 2: Finish[2024 年諾貝爾物理學獎得主是 John Hopfield 和 Geoffrey Hinton]
```

### 可用的 Action 類型

```python
tools = {
    "Search": lambda query: web_search(query),
    "Calculator": lambda expr: eval(expr),
    "WikipediaSearch": lambda topic: wikipedia_search(topic),
    "GetCurrentDate": lambda _: datetime.now().strftime("%Y-%m-%d"),
    "Finish": lambda answer: answer
}
```

### ReAct Prompt 範本

```python
REACT_PROMPT = """
你是一個問題解決助手。你可以使用以下工具：

1. Search[query]: 在網路上搜尋資訊
2. Calculator[expression]: 計算數學表達式
3. Finish[answer]: 當你準備好給出最終答案時使用

你應該按照以下格式回應：

Thought: 你的思考過程
Action: 工具名稱[輸入]

範例：
問題：台灣最高的山是什麼？海拔多少公尺？

Thought: 我需要先找出台灣最高的山
Action: Search[台灣最高的山]
Observation: 玉山是台灣最高的山，海拔 3952 公尺

Thought: 我已經找到答案了
Action: Finish[台灣最高的山是玉山，海拔 3952 公尺]

現在開始：
問題：{question}

Thought:"""
```

### ReAct vs CoT

| 特性 | CoT | ReAct |
|------|-----|-------|
| 推理方式 | 純思考 | 思考 + 行動 |
| 資訊來源 | 模型內部知識 | 可獲取外部資訊 |
| 適用任務 | 邏輯推理、數學 | 需要即時資訊的任務 |
| 複雜度 | 低 | 高 |
| 工具需求 | 不需要 | 需要外部工具 |

### ReAct 的優勢
1. **即時資訊**：可以獲取最新資料
2. **突破知識限制**：不受訓練資料的限制
3. **可驗證**：行動結果可以驗證
4. **減少幻覺**：基於實際資料而非憑空生成

### ReAct 的挑戰
1. **工具依賴**：需要可靠的外部工具
2. **成本較高**：多次 LLM 呼叫
3. **錯誤累積**：一步錯誤可能導致後續錯誤
4. **行動選擇**：如何選擇正確的行動？

### 完整實作範例

```python
import re
from typing import Dict, Callable

class ReActAgent:
    def __init__(self, llm, tools: Dict[str, Callable]):
        self.llm = llm
        self.tools = tools
    
    def run(self, question: str, max_iterations: int = 10) -> str:
        """執行 ReAct 流程"""
        
        prompt = self.build_initial_prompt(question)
        
        for i in range(max_iterations):
            # 生成 Thought 和 Action
            response = self.llm.generate(prompt)
            
            # 解析回應
            thought, action, action_input = self.parse_action(response)
            
            print(f"\nThought {i+1}: {thought}")
            print(f"Action {i+1}: {action}[{action_input}]")
            
            # 檢查是否完成
            if action.lower() == "finish":
                return action_input
            
            # 執行 Action
            try:
                observation = self.tools[action](action_input)
            except KeyError:
                observation = f"錯誤：未知的工具 '{action}'"
            except Exception as e:
                observation = f"錯誤：{str(e)}"
            
            print(f"Observation {i+1}: {observation}")
            
            # 更新 prompt
            prompt += f"\n{response}"
            prompt += f"\nObservation: {observation}\n\nThought:"
        
        return "達到最大迭代次數，無法得出答案"
    
    def parse_action(self, response: str):
        """解析 LLM 回應，提取 thought, action 和 action_input"""
        
        # 提取 Thought
        thought_match = re.search(r'Thought:?\s*(.+?)(?=\nAction:|$)', 
                                  response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # 提取 Action
        action_match = re.search(r'Action:?\s*(\w+)\[(.+?)\]', response)
        if action_match:
            action = action_match.group(1)
            action_input = action_match.group(2)
        else:
            action = "Finish"
            action_input = response
        
        return thought, action, action_input
    
    def build_initial_prompt(self, question: str) -> str:
        """建立初始 prompt"""
        tools_desc = "\n".join([f"- {name}: {func.__doc__ or ''}" 
                                for name, func in self.tools.items()])
        
        return f"""
回答以下問題。你可以使用這些工具：

{tools_desc}

使用以下格式：
Thought: [你的思考]
Action: [工具名稱][輸入]

當你準備好答案時：
Action: Finish[你的最終答案]

問題：{question}

Thought:"""

# 使用範例
def search(query):
    """在網路上搜尋"""
    # 這裡應該呼叫實際的搜尋 API
    return f"搜尋結果：{query}"

def calculator(expression):
    """計算數學表達式"""
    try:
        return str(eval(expression))
    except:
        return "計算錯誤"

tools = {
    "Search": search,
    "Calculator": calculator,
    "Finish": lambda x: x
}

agent = ReActAgent(llm, tools)
answer = agent.run("台北到高雄的距離是多少？開車大約需要多久？")
```

### ReAct 的應用
- 問答系統（需要查詢資料）
- 資料分析（需要計算）
- 任務自動化（需要執行操作）
- 研究助手（需要查閱多個來源）

## CoT vs ReAct 的選擇

### 使用 CoT
- 問題主要是邏輯推理
- 不需要外部資訊
- 希望看到詳細的推理過程
- 成本考量（較便宜）

**範例任務**：
- 數學證明
- 邏輯謎題
- 策略遊戲分析

### 使用 ReAct
- 需要即時或外部資訊
- 需要執行計算或操作
- 任務需要多步驟的資訊獲取
- 需要驗證資訊的真實性

**範例任務**：
- "今天的天氣如何？"
- "計算複雜的數學表達式"
- "查詢最新的新聞"

### 組合使用
可以結合 CoT 和 ReAct 的優點：

```python
def cot_react_agent(question):
    """結合 CoT 和 ReAct"""
    
    # 先用 CoT 分析問題
    analysis = cot_analysis(question)
    
    # 根據分析決定是否需要外部資訊
    if need_external_info(analysis):
        # 使用 ReAct 獲取資訊
        return react_agent.run(question)
    else:
        # 使用純 CoT 推理
        return cot_agent.solve(question)
```

## 其他相關技術

### 1. Plan-and-Solve
先規劃整體策略，再逐步執行。

### 2. Least-to-Most Prompting
從簡單的子問題開始，逐步解決更複雜的問題。

### 3. Self-Ask
模型自己提出子問題並回答。

## 最佳實踐

1. **明確指示**：清楚告訴模型要逐步思考
2. **提供範例**：Few-shot 通常比 Zero-shot 效果好
3. **驗證機制**：加入驗證步驟確保答案正確
4. **錯誤處理**：準備處理工具呼叫失敗的情況
5. **成本控制**：注意 Token 使用和 API 呼叫次數

## 總結

CoT 和 ReAct 是提升 LLM 推理能力的重要技術：

- **CoT**：讓模型「思考」得更深入
- **ReAct**：讓模型能夠「行動」獲取資訊

兩者結合使用可以打造出更強大、更可靠的 AI Agent。

