# AI-Mock-Interview

面向计算机相关专业学生的 `AI 模拟面试与能力提升平台` 项目仓库。

当前阶段已完成的重点是 `Java 后端岗位 RAG 基座`，用于支撑以下能力：

- 模拟面试出题
- 标准回答检索
- 追问生成
- 回答评估
- 面试反馈生成

## 当前进展

已完成：

- Java 后端主知识库标准化
- Java 后端主知识库扩充至 100 题
- 主库与补充库格式规范文档
- 岗位化入库脚本 `scripts/ingest_data.py`
- 检索验证脚本 `scripts/query_test.py`
- 最小文本面试流程引擎 `scripts/interview_flow.py`
- 最小 HTTP 面试接口 `scripts/interview_api.py`

进行中：

- Java RAG 基座收尾
- 多岗位知识库扩展方案
- 面试流程与接口层打通

待完成：

- 面试评估与报告生成
- 语音链路
- 前端交互界面

## 目录结构

```text
AI_Interview_Project/
├─ data/
│  ├─ java_backend/
│  │  ├─ Java后端主知识库-标准化面试题库.md
│  │  ├─ Java后端主知识库格式规范.md
│  │  └─ Java后端补充知识库格式规范.md
│  └─ web_frontend/
├─ docs/
│  ├─ 项目实施路线图.md
│  ├─ Java岗位RAG使用说明.md
│  ├─ 知识库与目录规划.md
│  ├─ 最小面试流程引擎说明.md
│  ├─ 最小面试接口说明.md
│  ├─ 前端报告页对接说明.md
│  └─ 最小前端页面说明.md
├─ scripts/
│  ├─ ingest_data.py
│  ├─ query_test.py
│  ├─ interview_flow.py
│  └─ interview_api.py
├─ webapp/
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
├─ models/
├─ db/
└─ README.md
```

说明：

- `data/` 存放岗位化知识库
- `docs/` 存放项目文档、实施说明与扩展规范
- `scripts/` 存放入库、检索、验证等脚本
- `models/` 存放本地 embedding 模型，不提交仓库
- `db/` 存放本地向量库，不提交仓库

## Java RAG 基座

Java 岗位当前采用：

- 主知识库：结构化单题标准库
- 补充知识库：专题原理、案例、排障、优化资料
- 向量库：Chroma
- Embedding 模型：`bge-small-zh-v1.5`

主知识库当前固定结构为：

- 元数据
- 面试题
- 标准回答
- 追问点
- 评分点
- 常见失分点

详细使用方法见：

- [Java岗位RAG使用说明.md](D:\AI_Interview_Project\docs\Java岗位RAG使用说明.md)
- [最小面试流程引擎说明.md](D:\AI_Interview_Project\docs\最小面试流程引擎说明.md)
- [最小面试接口说明.md](D:\AI_Interview_Project\docs\最小面试接口说明.md)
- [前端报告页对接说明.md](D:\AI_Interview_Project\docs\前端报告页对接说明.md)
- [最小前端页面说明.md](D:\AI_Interview_Project\docs\最小前端页面说明.md)

## 常用命令

### 1. 主知识库入库

```powershell
D:\AI_Interview_Project\scripts\.venv\Scripts\python.exe D:\AI_Interview_Project\scripts\ingest_data.py `
  --data-dir D:\AI_Interview_Project\data `
  --db-dir D:\AI_Interview_Project\db `
  --collection-name java_interview_main `
  --files java_backend/Java后端主知识库-标准化面试题库.md `
  --embedding-model D:\AI_Interview_Project\models\bge-small-zh-v1.5-full `
  --local-model-only `
  --clear-collection
```

### 2. 检索验证

```powershell
D:\AI_Interview_Project\scripts\.venv\Scripts\python.exe D:\AI_Interview_Project\scripts\query_test.py `
  --db-dir D:\AI_Interview_Project\db `
  --collection-name java_interview_main `
  --embedding-model D:\AI_Interview_Project\models\bge-small-zh-v1.5-full `
  --local-model-only `
  --role java_backend `
  --query "请解释 HashMap 的底层原理"
```

### 3. 启动最小面试接口

```powershell
D:\AI_Interview_Project\scripts\.venv\Scripts\python.exe D:\AI_Interview_Project\scripts\interview_api.py `
  --host 127.0.0.1 `
  --port 8010
```

启动后访问：

- [http://127.0.0.1:8010/app](http://127.0.0.1:8010/app)

### 4. 启用 LLM 面试官评价

复制 `.env.example` 为 `.env`，填入真实模型配置：

```powershell
Copy-Item .env.example .env
```

```text
LLM_JUDGE_ENABLED=1
LLM_JUDGE_API_URL=https://api.deepseek.com/chat/completions
LLM_JUDGE_MODEL=deepseek-v4-pro
LLM_JUDGE_MODE=full
DEEPSEEK_API_KEY=你的真实密钥
```

然后正常启动接口即可。也可以不用 `.env`，直接通过启动参数传入：

```powershell
python scripts\interview_api.py `
  --host 127.0.0.1 `
  --port 8010 `
  --llm-judge-enabled `
  --llm-judge-api-url https://api.deepseek.com/chat/completions `
  --llm-judge-model deepseek-v4-pro `
  --llm-judge-mode full `
  --llm-judge-api-key-env DEEPSEEK_API_KEY
```

LLM 接入状态可查看：

- [http://127.0.0.1:8010/llm/status](http://127.0.0.1:8010/llm/status)

`LLM_JUDGE_MODE` 支持：

- `conservative`：贴近规则评分，默认最多偏离 1 分
- `balanced`：允许 LLM 修正规则遗漏，默认最多偏离 2.5 分
- `full`：当前推荐，LLM 主导评价，规则评分作为证据底座和失败兜底

## 当前注意事项

- 主知识库入库时必须使用 `--files java_backend/Java后端主知识库-标准化面试题库.md`
- 不要把格式规范文档混入主 collection
- 主库与补充库应分层管理，不建议混在同一个 collection
- PowerShell 终端中文显示可能乱码，分析内容时优先使用 `python -X utf8`
- `langchain_community.vectorstores.Chroma` 当前有 deprecation warning，但暂不影响本阶段演示

## 下一步开发顺序

建议严格按下面顺序推进：

1. 收尾 Java RAG 基座
2. 补齐第二岗位知识库
3. 实现面试流程引擎
4. 实现评估与反馈引擎
5. 接入语音与多模态
6. 补齐前端界面与比赛材料

完整路线见：

- [项目实施路线图.md](D:\AI_Interview_Project\docs\项目实施路线图.md)

## 仓库提交建议

建议提交：

- `data/java_backend/*.md`
- `docs/*.md`
- `scripts/ingest_data.py`
- `scripts/query_test.py`
- `README.md`

不要提交：

- `models/`
- `db/`
- `db_probe/`
- `scripts/.venv/`
- 临时测试数据库和缓存文件
