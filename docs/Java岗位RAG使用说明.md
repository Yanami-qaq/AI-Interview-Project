# Java岗位 RAG 使用说明

## 1. 目标

本模块用于支撑 `Java 后端岗位` 的：

- 模拟面试出题
- 标准回答检索
- 动态追问生成
- 回答评估
- 面试反馈生成

当前重点是先把 `Java 岗位 RAG 基座` 做稳定，再接入完整平台。

---

## 2. 当前组成

Java 岗位知识库由两层组成：

### 主知识库

文件：

- `D:\AI_Interview_Project\data\java_backend\Java后端主知识库-标准化面试题库.md`

作用：

- 负责标准化面试题
- 负责标准回答模板
- 负责追问点
- 负责评分点
- 负责常见失分点

### 补充知识库

作用：

- 提供主库之外的原理细节
- 提供工程实践案例
- 提供排障、优化、设计类背景材料

格式规范：

- [Java后端补充知识库格式规范.md](D:\AI_Interview_Project\data\java_backend\Java后端补充知识库格式规范.md)

---

## 3. 主知识库固定格式

每道题必须采用以下结构：

- 元数据
- 面试题
- 标准回答
- 追问点
- 评分点
- 常见失分点

题号统一使用：

- `## Q001`
- `## Q002`
- `## Q003`

一直递增。

格式规范见：

- [Java后端主知识库格式规范.md](D:\AI_Interview_Project\data\java_backend\Java后端主知识库格式规范.md)

---

## 4. 入库脚本能力

脚本：

- `D:\AI_Interview_Project\scripts\ingest_data.py`

当前已支持：

- 命令行参数
- 自定义 collection 名称
- 指定本地 embedding 模型
- `--local-model-only`
- `--files` 指定单文件入库
- 自动抽取 metadata
- 对标准化题库按 `## Qxxx` 单题切分
- 单题级 metadata 绑定

主知识库会优先抽取这些字段：

- `job_role`
- `source`
- `source_path`
- `title`
- `topic`
- `question_type`
- `difficulty`
- `keywords`
- `question`
- `section`

---

## 5. 检索脚本能力

脚本：

- `D:\AI_Interview_Project\scripts\query_test.py`

当前支持：

- 指定 collection
- 指定本地 embedding 模型
- `--local-model-only`
- `role` 过滤
- `topic` 过滤
- 输出检索结果 metadata

当前结果会输出：

- `title`
- `topic`
- `question_type`
- `difficulty`
- `section`
- `source`
- `question`

---

## 6. 推荐模型与参数

当前统一使用本地 embedding 模型：

- `D:\AI_Interview_Project\models\bge-small-zh-v1.5-full`

推荐统一参数：

- `--embedding-model D:\AI_Interview_Project\models\bge-small-zh-v1.5-full`
- `--local-model-only`

这样可以保证：

- 演示时离线可用
- embedding 一致
- 避免线上下载依赖

---

## 7. 推荐入库命令

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

注意：

- 必须加 `--files java_backend/Java后端主知识库-标准化面试题库.md`
- 否则可能把规范文档一起入库，污染主 collection

---

## 8. 推荐检索验证命令

```powershell
D:\AI_Interview_Project\scripts\.venv\Scripts\python.exe D:\AI_Interview_Project\scripts\query_test.py `
  --db-dir D:\AI_Interview_Project\db `
  --collection-name java_interview_main `
  --embedding-model D:\AI_Interview_Project\models\bge-small-zh-v1.5-full `
  --local-model-only `
  --role java_backend `
  --query "请解释 HashMap 的底层原理"
```

---

## 9. 当前 collection 规划

建议至少分成两类：

### 主库 collection

示例：

- `java_interview_main`

用途：

- 出题
- 标准答案检索
- 评分点与失分点参考

### 补充库 collection

示例：

- `java_interview_support`

用途：

- 原理细化
- 案例补充
- 工程化回答增强

不建议现阶段把主库和补充库混在同一个 collection。

---

## 10. 后续平台接入建议

后续面试流程中，可以把 Java 岗位 RAG 拆成三类能力：

### 出题检索

从主库按岗位、主题、难度、题型检索候选题。

### 追问增强

先从主库拿追问点，再结合补充库扩展技术细节。

### 评估参考

基于主库中的标准回答、评分点、常见失分点做评估 Prompt 的参考输入。

---

## 11. 当前已知问题

- 本地 Chroma 在部分路径可能出现 `disk I/O error`
- PowerShell 中文输出可能乱码
- Chroma 当前依赖存在 deprecation warning

说明：

- `disk I/O error` 更偏运行环境问题，不完全是脚本逻辑问题
- 演示时建议优先使用确认可正常写库的目录

---

## 12. 下一步建议

Java RAG 基座完成后，建议按下面顺序继续：

1. 规划 Java 补充知识库分层
2. 新增第二岗位主知识库
3. 抽象统一的岗位知识库目录结构
4. 接入面试流程引擎
5. 接入评估引擎
