# AI Agent 能力

## 什麼是 AI Agent？

AI Agent（AI 代理）是能夠自主感知環境、做出決策並採取行動以達成目標的系統。在 LLM 背景下，Agent 是能夠使用工具、規劃任務、反思結果的智能系統。

### Agent 的基本架構

```
環境 (Environment)
    ↓ 感知 (Perception)
Agent
    ↓ 決策 (Decision)
行動 (Action)
    ↓ 影響
環境 (Environment)
```

## Reflection（反思）

### 定義
Reflection 是 Agent 對自己的輸出或行為進行評估和改進的能力，類似人類的自我反思過程。

### 為什麼需要 Reflection？
1. **提高品質**：透過自我批評找出問題並改進
2. **減少錯誤**：及早發現並修正錯誤
3. **迭代改進**：逐步提升輸出品質

### Reflection 的流程

```
1. 生成初始回應
2. 評估回應品質（找出問題）
3. 根據評估改進回應
4. 重複 2-3 直到滿意
```

### 實作範例

```python
def agent_with_reflection(task, max_iterations=3):
    """具有反思能力的 Agent"""
    
    # 第一次生成
    response = llm.generate(task)
    
    for i in range(max_iterations):
        # 反思：評估當前回應
        reflection_prompt = f"""
        請評估以下回應的品質：
        任務：{task}
        回應：{response}
        
        請指出：
        1. 回應是否完整？
        2. 有哪些錯誤或不足？
        3. 如何改進？
        """
        
        reflection = llm.generate(reflection_prompt)
        
        # 如果評估認為已經很好，就結束
        if "沒有問題" in reflection or "已經很好" in reflection:
            break
        
        # 根據反思改進回應
        improve_prompt = f"""
        原始任務：{task}
        當前回應：{response}
        評估意見：{reflection}
        
        請根據評估意見改進回應。
        """
        
        response = llm.generate(improve_prompt)
    
    return response
```

### Reflection 的變體

#### 1. Self-Refine
Agent 自己評估和改進輸出。

#### 2. Reflexion
結合長期記憶，從過去的錯誤中學習。

```python
class ReflexionAgent:
    def __init__(self):
        self.memory = []  # 儲存過去的經驗
    
    def act(self, task):
        # 參考過去的經驗
        past_mistakes = self.get_relevant_mistakes(task)
        
        # 生成回應時考慮過去的錯誤
        response = self.generate_with_history(task, past_mistakes)
        
        # 評估並記錄
        if self.evaluate(response):
            return response
        else:
            self.memory.append({
                "task": task,
                "failed_response": response,
                "reason": self.analyze_failure(response)
            })
            return self.retry_with_reflection(task)
```

#### 3. Chain of Verification (CoVe)
生成回應後，生成驗證問題來檢查答案的正確性。

### 應用場景
- 內容創作（反覆修改直到滿意）
- 程式碼生成（檢查並修正錯誤）
- 複雜問題解決（迭代改進方案）

## Tool Use（工具使用）

### 定義
Tool Use 是 Agent 呼叫外部工具或 API 來擴展自身能力的機制，彌補 LLM 的限制。

### 為什麼需要 Tool Use？
LLM 的限制：
1. **知識截止**：訓練資料有時間限制
2. **數學計算**：對複雜計算不準確
3. **實時資訊**：無法獲取即時資料
4. **外部操作**：無法直接操作系統、資料庫等

### 常見的工具類型

#### 1. 搜尋工具
```python
def search_web(query):
    """搜尋網路獲取最新資訊"""
    # 呼叫搜尋 API（如 Google Search, Bing Search）
    results = google_search_api(query)
    return results

# 範例
search_web("2024 年奧運金牌榜")
```

#### 2. 計算工具
```python
def calculator(expression):
    """執行數學計算"""
    try:
        result = eval(expression)
        return f"計算結果：{result}"
    except:
        return "計算錯誤"

# 範例
calculator("(123 + 456) * 789 / 12")
```

#### 3. 資料庫查詢
```python
def query_database(sql):
    """查詢資料庫"""
    import sqlite3
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    return results
```

#### 4. API 呼叫
```python
def get_weather(city):
    """獲取天氣資訊"""
    api_url = f"https://api.weather.com/{city}"
    response = requests.get(api_url)
    return response.json()
```

#### 5. 程式碼執行
```python
def execute_python_code(code):
    """執行 Python 程式碼"""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return exec_globals
    except Exception as e:
        return f"執行錯誤：{str(e)}"
```

### Function Calling（函式呼叫）

OpenAI 提供的 Function Calling 功能：

```python
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "獲取指定城市的天氣",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名稱"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "台北今天天氣如何？"}],
    tools=tools,
    tool_choice="auto"
)

# Agent 會決定是否呼叫工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # 執行實際的函式
    result = get_weather(arguments['city'])
    
    # 將結果回傳給模型
    final_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "台北今天天氣如何？"},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            }
        ]
    )
```

### Tool Use 的挑戰
1. **工具選擇**：如何選擇正確的工具？
2. **參數生成**：如何生成正確的參數？
3. **錯誤處理**：工具呼叫失敗怎麼辦？
4. **安全性**：如何防止惡意的工具呼叫？

## Planning（規劃）

### 定義
Planning 是 Agent 為達成複雜目標而制定行動序列的能力。

### 為什麼需要 Planning？
- 複雜任務需要分解成多個步驟
- 需要考慮依賴關係和順序
- 提高執行效率和成功率

### Planning 的類型

#### 1. Single-path Planning（單路徑規劃）
生成一個固定的步驟序列。

```python
def plan_task(goal):
    """為目標生成計畫"""
    prompt = f"""
    目標：{goal}
    
    請制定詳細的執行計畫，包含：
    1. 需要完成的步驟
    2. 每個步驟的具體行動
    3. 預期的結果
    
    請以 JSON 格式輸出：
    [
        {{"step": 1, "action": "...", "expected_result": "..."}},
        ...
    ]
    """
    
    plan = llm.generate(prompt)
    return json.loads(plan)
```

#### 2. Multi-path Planning（多路徑規劃）
考慮多種可能的路徑，選擇最佳方案。

#### 3. Adaptive Planning（自適應規劃）
根據執行結果動態調整計畫。

```python
class AdaptivePlanner:
    def execute_plan(self, goal):
        plan = self.create_initial_plan(goal)
        
        for step in plan:
            result = self.execute_step(step)
            
            # 檢查結果
            if not self.is_successful(result):
                # 重新規劃
                plan = self.replan(goal, step, result)
            
            if self.is_goal_achieved(goal):
                break
        
        return result
```

### Planning 策略

#### 1. Forward Planning（前向規劃）
從當前狀態開始，逐步推進到目標。

```
當前狀態 → 行動1 → 狀態1 → 行動2 → 狀態2 → 目標
```

#### 2. Backward Planning（反向規劃）
從目標開始，反推需要什麼前置條件。

```
目標 → 需要狀態2 → 需要行動2 → 需要狀態1 → 需要行動1 → 當前狀態
```

#### 3. Hierarchical Planning（層級規劃）
將任務分解成多層子任務。

```
總目標
├─ 子目標1
│  ├─ 步驟1.1
│  └─ 步驟1.2
└─ 子目標2
   ├─ 步驟2.1
   └─ 步驟2.2
```

### 實作範例：任務規劃 Agent

```python
class PlanningAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def plan_and_execute(self, goal):
        # 1. 分解任務
        subtasks = self.decompose_task(goal)
        
        # 2. 為每個子任務規劃
        plan = []
        for subtask in subtasks:
            steps = self.plan_subtask(subtask)
            plan.extend(steps)
        
        # 3. 執行計畫
        results = []
        for step in plan:
            result = self.execute_step(step)
            results.append(result)
            
            # 檢查是否需要重新規劃
            if self.need_replan(result):
                remaining_plan = self.replan(goal, results)
                plan = remaining_plan
        
        return results
    
    def decompose_task(self, goal):
        """將目標分解成子任務"""
        prompt = f"將以下目標分解成可執行的子任務：{goal}"
        return self.llm.generate(prompt)
```

### Planning 的應用
- 複雜任務自動化
- 多步驟推理
- 專案管理
- 遊戲 AI

## Multi-agent（多代理系統）

### 定義
Multi-agent 系統是多個 AI Agent 協作或競爭以完成任務的系統。

### 為什麼需要 Multi-agent？
1. **專業分工**：不同 Agent 擅長不同任務
2. **平行處理**：多個 Agent 同時工作
3. **觀點多樣性**：不同 Agent 提供不同視角
4. **複雜問題**：單一 Agent 難以處理的問題

### Multi-agent 的架構模式

#### 1. 協作模式（Collaborative）
Agent 們共同合作達成共同目標。

```python
class CollaborativeSystem:
    def __init__(self):
        self.researcher = Agent(role="研究員")
        self.analyst = Agent(role="分析師")
        self.writer = Agent(role="寫手")
    
    def generate_report(self, topic):
        # 研究員收集資料
        data = self.researcher.research(topic)
        
        # 分析師分析資料
        insights = self.analyst.analyze(data)
        
        # 寫手撰寫報告
        report = self.writer.write(insights)
        
        return report
```

#### 2. 競爭模式（Competitive）
Agent 們競爭或辯論，選出最佳方案。

```python
class DebateSystem:
    def __init__(self):
        self.agents = [
            Agent(role="正方"),
            Agent(role="反方"),
            Agent(role="裁判")
        ]
    
    def debate(self, topic):
        # 正方提出論點
        pro_argument = self.agents[0].argue(topic, stance="pro")
        
        # 反方反駁
        con_argument = self.agents[1].argue(topic, stance="con", 
                                             opponent=pro_argument)
        
        # 裁判評判
        decision = self.agents[2].judge(pro_argument, con_argument)
        
        return decision
```

#### 3. 層級模式（Hierarchical）
有管理者 Agent 協調其他 Agent。

```python
class HierarchicalSystem:
    def __init__(self):
        self.manager = Agent(role="管理者")
        self.workers = [
            Agent(role="工作者1"),
            Agent(role="工作者2"),
            Agent(role="工作者3")
        ]
    
    def execute_task(self, task):
        # 管理者分配任務
        subtasks = self.manager.delegate(task, self.workers)
        
        # 工作者執行
        results = []
        for worker, subtask in zip(self.workers, subtasks):
            result = worker.execute(subtask)
            results.append(result)
        
        # 管理者整合結果
        final_result = self.manager.integrate(results)
        
        return final_result
```

### Multi-agent 通訊機制

#### 1. 直接通訊
Agent 之間直接交換訊息。

```python
class Agent:
    def send_message(self, receiver, message):
        receiver.receive_message(self, message)
    
    def receive_message(self, sender, message):
        response = self.process_message(message)
        self.send_message(sender, response)
```

#### 2. 廣播通訊
Agent 向所有其他 Agent 廣播訊息。

```python
class MessageBus:
    def __init__(self):
        self.agents = []
    
    def broadcast(self, sender, message):
        for agent in self.agents:
            if agent != sender:
                agent.receive_message(sender, message)
```

#### 3. 黑板系統（Blackboard）
共享的知識空間，Agent 可以讀寫。

```python
class Blackboard:
    def __init__(self):
        self.knowledge = {}
    
    def write(self, key, value):
        self.knowledge[key] = value
    
    def read(self, key):
        return self.knowledge.get(key)

# Agent 透過黑板共享資訊
blackboard = Blackboard()
agent1.write_to_blackboard(blackboard, "data", value)
agent2_data = agent2.read_from_blackboard(blackboard, "data")
```

### Multi-agent 的挑戰

1. **協調成本**：Agent 之間的溝通和協調需要成本
2. **一致性**：如何確保 Agent 們的行動一致？
3. **衝突解決**：Agent 之間有衝突時如何處理？
4. **計算成本**：多個 Agent 意味著多次 LLM 呼叫

### 實作範例：軟體開發團隊

```python
class SoftwareDevTeam:
    def __init__(self):
        self.pm = Agent(role="專案經理", 
                       skills=["需求分析", "任務分配"])
        self.dev = Agent(role="開發者", 
                        skills=["編寫程式", "除錯"])
        self.qa = Agent(role="測試人員", 
                       skills=["測試", "找bug"])
    
    def develop_feature(self, requirement):
        # PM 分析需求並規劃
        plan = self.pm.analyze_and_plan(requirement)
        
        # 開發者實作
        code = self.dev.implement(plan)
        
        # QA 測試
        test_result = self.qa.test(code)
        
        # 如果有 bug，開發者修復
        while not test_result.passed:
            bugs = test_result.bugs
            code = self.dev.fix_bugs(code, bugs)
            test_result = self.qa.test(code)
        
        # PM 驗收
        if self.pm.review(code, requirement):
            return code
        else:
            # 重新開發
            return self.develop_feature(requirement)
```

### Multi-agent 框架
- **AutoGen**：微軟的 Multi-agent 框架
- **CrewAI**：基於角色的協作框架
- **LangGraph**：支援複雜的 Agent 互動流程

## 四種能力的組合

在實際應用中，這四種能力常常組合使用：

```python
class AdvancedAgent:
    def solve_complex_task(self, task):
        # 1. Planning：制定計畫
        plan = self.plan(task)
        
        results = []
        for step in plan:
            # 2. Tool Use：使用工具執行步驟
            result = self.execute_with_tools(step)
            
            # 3. Reflection：評估結果
            evaluation = self.reflect(result, step)
            
            if not evaluation.is_good:
                # 重新執行或調整計畫
                result = self.improve(result, evaluation)
            
            results.append(result)
        
        # 4. Multi-agent：必要時諮詢其他 Agent
        if self.need_expert_opinion(results):
            expert_agent = self.get_expert()
            results = expert_agent.review_and_improve(results)
        
        return self.synthesize(results)
```

## 總結

| 能力 | 核心價值 | 主要用途 | 複雜度 |
|------|---------|---------|--------|
| Reflection | 自我改進 | 提高輸出品質 | 中等 |
| Tool Use | 擴展能力 | 彌補 LLM 限制 | 中等 |
| Planning | 任務分解 | 處理複雜任務 | 高 |
| Multi-agent | 協作分工 | 專業化與平行處理 | 非常高 |

這四種能力讓 AI Agent 從簡單的問答系統進化為能夠自主完成複雜任務的智能系統。

