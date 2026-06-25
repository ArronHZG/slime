# slime — 详细技术 Wiki

> **slime** 是一个面向强化学习规模扩展的 LLM 后训练框架，核心目标：高性能训练 + 灵活数据生成。  
> 官方文档：https://thudm.github.io/slime/ | 版本：0.3.0

---

## 目录

1. [项目概览](#1-项目概览)
2. [高层架构](#2-高层架构)
3. [核心模块详解](#3-核心模块详解)
   - [Training 模块（Megatron）](#31-training-模块megatron)
   - [Rollout 模块（SGLang）](#32-rollout-模块sglang)
   - [Data Buffer](#33-data-buffer)
   - [Ray 编排层](#34-ray-编排层)
4. [训练主循环](#4-训练主循环)
5. [核心数据类型](#5-核心数据类型)
6. [定制化接口](#6-定制化接口)
7. [部署模式](#7-部署模式)
8. [高级特性](#8-高级特性)
9. [插件系统](#9-插件系统)
10. [生态项目](#10-生态项目)
11. [配置指南](#11-配置指南)
12. [测试与 CI](#12-测试与-ci)
13. [目录结构](#13-目录结构)

---

## 1. 项目概览

slime 由清华大学 THUDM 实验室开发，是 GLM-5、GLM-4.x 等前沿模型 RL 后训练的核心基础设施。

### 设计哲学

```
correctness-first  →  native pass-through  →  lightweight & opinionated
```

| 原则 | 说明 |
|------|------|
| **正确性优先** | RL bug 往往是无声的。slime 保持数据流显式可见，支持独立的 rollout-only / train-only 调试路径 |
| **原生直通** | Megatron 参数直接传递；SGLang 参数加 `--sglang-` 前缀即可使用 |
| **轻量且专注** | 深度优化 Megatron + SGLang 这一路径，不做多引擎最大公约数抽象 |
| **最大数据生成自由度** | 数学/代码/搜索/工具/沙盒/验证器/多智能体均通过数据生成接口插入，不修改训练核 |

### 已验证模型

- **GLM 系列**：GLM-5.2, 5.1, 5, 4.7, 4.6, 4.5
- **Qwen 系列**：Qwen3.6, 3.5, 3Next, 3MoE, 3, 2.5
- **DeepSeek 系列**：V3, V3.1, R1
- **Llama 3**

---

## 2. 高层架构

### 2.1 系统全局架构

```mermaid
graph TB
    subgraph EntryPoint["入口点"]
        TP[train.py<br/>同步训练]
        TAP[train_async.py<br/>异步训练]
    end

    subgraph RayLayer["Ray 编排层"]
        PG[placement_group.py<br/>GPU 资源分配]
        RM[RolloutManager<br/>ray/rollout.py]
        AG[ActorGroup<br/>ray/actor_group.py]
        TA[TrainingActor<br/>ray/train_actor.py]
    end

    subgraph TrainModule["Training 模块"]
        direction TB
        MEG[Megatron-LM<br/>分布式训练]
        ACT[Actor Model<br/>策略网络]
        CRI[Critic Model<br/>价值网络（可选）]
        CKPT[Checkpoint Manager]
        LOSS[Loss Computation<br/>PPO/GRPO/SFT]
    end

    subgraph RolloutModule["Rollout 模块"]
        direction TB
        SGL[SGLang Engine<br/>高性能推理]
        RTR[SGLang Router<br/>负载均衡]
        ROLL[sglang_rollout.py<br/>主 rollout 逻辑]
        ASYNC[fully_async_rollout.py<br/>异步 rollout]
        RMH[Reward Model Hub<br/>rm_hub/]
        FH[Filter Hub<br/>filter_hub/]
    end

    subgraph DataBuffer["Data Buffer"]
        DS[DataSource<br/>data_source.py]
        BUF[RolloutBuffer<br/>样本缓存]
        PROMPT[Prompt 初始化]
    end

    subgraph CustomHooks["定制化插件（可选）"]
        CGF[custom_generate_function]
        CRM[custom_rm]
        RFP[rollout_function_path]
        DSP[data_source_path]
    end

    TP --> PG
    TAP --> PG
    PG --> RM
    PG --> AG
    AG --> TA
    TA --> MEG
    MEG --> ACT
    MEG --> CRI
    ACT --> CKPT
    ACT --> LOSS

    RM --> SGL
    RM --> RTR
    RTR --> SGL
    RM --> ROLL
    ROLL --> RMH
    ROLL --> FH
    ROLL --> ASYNC

    ROLL --> DS
    DS --> BUF
    BUF --> PROMPT

    CGF -.-> ROLL
    CRM -.-> RMH
    RFP -.-> ROLL
    DSP -.-> DS

    DataBuffer <--> TrainModule
    DataBuffer <--> RolloutModule
    TrainModule -- "weight sync" --> RolloutModule

    style EntryPoint fill:#e8f4f8,stroke:#2196F3
    style RayLayer fill:#f3e5f5,stroke:#9C27B0
    style TrainModule fill:#e8f5e9,stroke:#4CAF50
    style RolloutModule fill:#fff3e0,stroke:#FF9800
    style DataBuffer fill:#fce4ec,stroke:#E91E63
    style CustomHooks fill:#f5f5f5,stroke:#9E9E9E,stroke-dasharray: 5 5
```

### 2.2 三大核心模块交互

```mermaid
sequenceDiagram
    participant T as Training<br/>(Megatron)
    participant DB as Data Buffer
    participant R as Rollout<br/>(SGLang)

    Note over T,R: 初始化阶段
    T->>R: 推送初始权重 (update_weights)
    T->>DB: 初始化 prompt 数据集

    loop 训练循环 (rollout_id = 0..N)
        DB->>R: 提供 prompts (get_samples)
        R->>R: 生成响应 (SGLang 推理)
        R->>R: 计算奖励 (reward model)
        R->>DB: 存储 samples (add_samples)
        DB->>T: 提供训练数据
        T->>T: 计算 loss, 更新参数
        T->>DB: 可选: 保存 checkpoint
        T->>R: 同步新权重 (weight sync)
    end
```

---

## 3. 核心模块详解

### 3.1 Training 模块（Megatron）

基于 Megatron-LM 实现分布式大模型训练，支持多种并行策略。

```mermaid
graph LR
    subgraph Parallelism["并行策略"]
        TP[Tensor Parallelism<br/>张量并行]
        PP[Pipeline Parallelism<br/>流水线并行]
        DP[Data Parallelism<br/>数据并行]
        CP[Context Parallelism<br/>上下文并行]
        EP[Expert Parallelism<br/>MoE 专家并行]
    end

    subgraph TrainComp["训练组件"]
        ACT[Actor<br/>策略网络]
        CRI[Critic<br/>价值网络]
        REF[Reference<br/>参考模型（KL）]
    end

    subgraph LossTypes["Loss 类型"]
        PPO[PPO Loss]
        GRPO[GRPO Loss]
        SFT[SFT Loss]
        CUSTOM[Custom Loss]
    end

    subgraph WeightSync["权重同步"]
        STD[Standard Checkpoint<br/>标准检查点]
        DELTA[Delta Weight Sync<br/>增量权重同步]
        DISK[Disk Transport<br/>磁盘传输]
    end

    Parallelism --> TrainComp
    TrainComp --> LossTypes
    TrainComp --> WeightSync
```

**关键文件**：

| 文件 | 说明 |
|------|------|
| `slime/backends/megatron_utils/actor.py` | Megatron 训练 Actor 实现 |
| `slime/backends/megatron_utils/loss.py` | PPO/GRPO/SFT loss 计算 |
| `slime/backends/megatron_utils/model.py` | 模型定义 |
| `slime/backends/megatron_utils/checkpoint.py` | checkpoint 管理 |
| `slime/backends/megatron_utils/data.py` | 数据加载（与 Data Buffer 对接） |
| `slime/utils/ppo_utils.py` | PPO 算法核心实现（729 行） |
| `slime/utils/mask_utils.py` | Token 级 loss mask 计算 |

### 3.2 Rollout 模块（SGLang）

基于 SGLang 实现高性能推理与数据生成。

```mermaid
graph TB
    subgraph RolloutTypes["Rollout 类型"]
        SYNC[sglang_rollout.py<br/>标准同步 rollout]
        STREAM[sglang_streaming_rollout.py<br/>流式 rollout]
        ASYNC[fully_async_rollout.py<br/>全异步 rollout（长尾任务）]
        SFT[sft_rollout.py<br/>SFT rollout]
        OPD[on_policy_distillation.py<br/>在线策略蒸馏]
    end

    subgraph RewardHub["奖励模型 Hub"]
        MATH[math.py<br/>数学验证]
        F1[f1.py<br/>F1 分数]
        GPQA[gpqa.py<br/>GPQA]
        DEEP[deepscaler.py<br/>DeepScaler]
        IFB[ifbench.py<br/>IFBench]
        REMOTE[remote_rm<br/>远程奖励服务]
        CUSTOM_RM[custom_rm<br/>用户自定义]
    end

    subgraph FilterHub["过滤器 Hub"]
        DSF[dynamic_sampling_filters.py<br/>动态采样过滤]
        BF[buffer_filter<br/>缓冲区过滤]
        RSF[rollout_sample_filter<br/>样本级过滤]
    end

    subgraph SGLangDeploy["SGLang 部署"]
        ENG[SGLangEngine<br/>推理引擎]
        RTR2[SGLang Router<br/>请求路由]
        PD[PD Disaggregation<br/>预填充/解码分离]
        SPEC[Speculative Decoding<br/>投机解码]
    end

    SYNC --> RewardHub
    SYNC --> FilterHub
    ASYNC --> RewardHub
    RolloutTypes --> SGLangDeploy
```

**关键文件**：

| 文件 | 说明 |
|------|------|
| `slime/rollout/sglang_rollout.py` | 主 rollout 实现（609 行） |
| `slime/rollout/fully_async_rollout.py` | 全异步 rollout（适合长尾 agent 任务） |
| `slime/rollout/data_source.py` | 数据源抽象层 |
| `slime/backends/sglang_utils/sglang_engine.py` | SGLang 引擎封装 |
| `slime/backends/sglang_utils/sglang_config.py` | YAML 拓扑配置 |
| `slime/backends/sglang_utils/server_control.py` | 服务器生命周期管理 |

### 3.3 Data Buffer

Data Buffer 是连接 Training 和 Rollout 的桥梁模块。

```mermaid
graph LR
    subgraph DataBuffer["Data Buffer (data_source.py)"]
        INIT[Prompt 初始化<br/>JSONL/Parquet 加载]
        BUF[样本缓冲区<br/>rollout_buffer]
        QUEUE[请求队列<br/>动态调度]
    end

    subgraph DataFlow["数据流"]
        TRAIN_DATA[训练数据集<br/>--train-prompt-data]
        EVAL_DATA[评估数据集<br/>--eval-prompt-data]
        CUSTOM_DS[自定义数据源<br/>--data-source-path]
    end

    TRAIN_DATA --> INIT
    EVAL_DATA --> INIT
    CUSTOM_DS -.-> INIT
    INIT --> BUF
    BUF --> QUEUE
```

**支持的数据格式**：
- JSONL（`.jsonl`）
- Parquet（`.parquet`）
- 支持 Hugging Face datasets 格式

### 3.4 Ray 编排层

Ray 负责整个系统的分布式资源管理和进程编排。

```mermaid
graph TB
    subgraph RayCluster["Ray 集群"]
        HEAD[Ray Head Node]
        W1[Worker Node 1<br/>Training GPUs]
        W2[Worker Node 2<br/>Rollout GPUs]
        W3[Worker Node N]
    end

    subgraph PlacementGroups["Placement Groups"]
        PG_T[Training PG<br/>Megatron Actors]
        PG_R[Rollout PG<br/>SGLang Engines]
    end

    subgraph Actors["Ray Actors"]
        RM_ACT[RolloutManager<br/>@ray.remote]
        AG_ACT[ActorGroup<br/>@ray.remote]
        TA_ACT[TrainingActor<br/>@ray.remote]
        SGL_ACT[SGLangEngine<br/>@ray.remote]
    end

    HEAD --> W1
    HEAD --> W2
    HEAD --> W3
    PG_T --> W1
    PG_R --> W2
    PG_T --> AG_ACT
    PG_T --> TA_ACT
    PG_R --> RM_ACT
    PG_R --> SGL_ACT
```

**关键文件**：

| 文件 | 说明 |
|------|------|
| `slime/ray/rollout.py` | Ray RolloutManager（1485 行，核心编排） |
| `slime/ray/placement_group.py` | GPU Placement Group 创建 |
| `slime/ray/actor_group.py` | Actor Group 管理 |
| `slime/ray/train_actor.py` | Megatron 训练 Actor |

---

## 4. 训练主循环

### 4.1 主循环流程

```mermaid
flowchart TD
    START([开始]) --> PARSE[parse_args]
    PARSE --> PG_CREATE[create_placement_groups<br/>分配 GPU 资源]
    PG_CREATE --> INIT_TRACK[init_tracking<br/>初始化 WandB/TensorBoard]
    INIT_TRACK --> CREATE_RM[create_rollout_manager<br/>初始化 SGLang 引擎]
    CREATE_RM --> CREATE_MODEL[create_training_models<br/>初始化 Actor/Critic]
    CREATE_MODEL --> UW[actor_model.update_weights<br/>推送初始权重到 rollout]

    UW --> LOOP_START{rollout_id 循环<br/>start..num_rollout}

    LOOP_START --> EVAL_CHECK{eval_interval<br/>且 rollout_id==0?}
    EVAL_CHECK -- 是 --> EVAL1[rollout_manager.eval]
    EVAL_CHECK -- 否 --> GEN
    EVAL1 --> GEN

    GEN[rollout_manager.generate<br/>生成 rollout 数据] --> OFFLOAD_CHECK{offload_rollout?}
    OFFLOAD_CHECK -- 是 --> OFFLOAD[rollout_manager.offload]
    OFFLOAD_CHECK -- 否 --> TRAIN_CHECK
    OFFLOAD --> TRAIN_CHECK

    TRAIN_CHECK{use_critic?} -- 否 --> ACTOR_TRAIN[actor_model.async_train]
    TRAIN_CHECK -- 是 --> CRITIC_TRAIN[critic_model.async_train<br/>获取 value_refs]
    CRITIC_TRAIN --> ACTOR_TRAIN2[actor_model.async_train<br/>with value_refs]

    ACTOR_TRAIN --> SAVE_CHECK{should_save?}
    ACTOR_TRAIN2 --> SAVE_CHECK

    SAVE_CHECK -- 是 --> SAVE[save checkpoint]
    SAVE_CHECK -- 否 --> OFFLOAD2

    SAVE --> OFFLOAD2[offload_train<br/>清理训练内存]
    OFFLOAD2 --> UW2[actor_model.update_weights<br/>同步新权重到 rollout]
    UW2 --> EVAL_PERIODIC{eval_interval<br/>周期到?}
    EVAL_PERIODIC -- 是 --> EVAL2[rollout_manager.eval]
    EVAL_PERIODIC -- 否 --> LOOP_END

    EVAL2 --> LOOP_END{还有 rollout?}
    LOOP_END -- 是 --> LOOP_START
    LOOP_END -- 否 --> DISPOSE[rollout_manager.dispose]
    DISPOSE --> FINISH([finish_tracking 结束])

    style START fill:#4CAF50,color:#fff
    style FINISH fill:#4CAF50,color:#fff
    style GEN fill:#FF9800,color:#fff
    style ACTOR_TRAIN fill:#2196F3,color:#fff
    style ACTOR_TRAIN2 fill:#2196F3,color:#fff
    style EVAL1 fill:#9C27B0,color:#fff
    style EVAL2 fill:#9C27B0,color:#fff
```

### 4.2 单次 Rollout 内部流程

```mermaid
flowchart TD
    SAMPLES[从 DataSource 获取 prompts] --> GROUP[按 group_size 分组]
    GROUP --> DISPATCH[分发到 SGLang 引擎]
    DISPATCH --> GEN[SGLang 并行生成]
    GEN --> REWARD[计算奖励<br/>rm_hub / custom_rm]
    REWARD --> FILTER[动态采样过滤<br/>dynamic_sampling_filter]
    FILTER --> ADVANTAGE[计算优势函数<br/>GRPO/PPO advantage]
    ADVANTAGE --> LOGPROB[计算 rollout log probs]
    LOGPROB --> POSTPROCESS[数据后处理<br/>rollout_data_postprocess]
    POSTPROCESS --> BUFFER[存入 Data Buffer]
    BUFFER --> TRAIN_ITER[Megatron 训练迭代]
```

---

## 5. 核心数据类型

### 5.1 Sample 类

`Sample` 是 slime 中最核心的数据结构，贯穿 rollout → training 全流程。

```mermaid
classDiagram
    class Sample {
        +int group_index
        +int index
        +int rollout_id
        +str|list prompt
        +list~int~ tokens
        +dict multimodal_inputs
        +str response
        +int response_length
        +str label
        +float|dict reward
        +list~int~ loss_mask
        +list~str~ weight_versions
        +list~float~ rollout_log_probs
        +list~int~ rollout_top_p_token_ids
        +list~int~ rollout_top_p_token_offsets
        +list rollout_routed_experts
        +bool remove_sample
        +list~float~ teacher_log_probs
        +Status status
        +dict metadata
        +str generate_function_path
        +str custom_rm_path
        +dict train_metadata
        +str session_id
        +float non_generation_time
        +SpecInfo spec_info
        +PrefixCacheInfo prefix_cache_info
        +append_response_tokens()
        +get_reward_value()
        +to_dict()
        +from_dict()
    }

    class Status {
        <<enumeration>>
        PENDING
        COMPLETED
        TRUNCATED
        ABORTED
        FAILED
    }

    class SpecInfo {
        +int spec_accept_token_num
        +int spec_draft_token_num
        +int spec_verify_ct
        +int completion_token_num
        +float spec_accept_rate
        +float spec_accept_length
        +add(meta_info)
    }

    class PrefixCacheInfo {
        +int cached_tokens
        +int total_prompt_tokens
        +float prefix_cache_hit_rate
        +add(meta_info)
    }

    Sample --> Status
    Sample --> SpecInfo
    Sample --> PrefixCacheInfo
```

**关键字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `group_index` | `int` | 在 group 内的索引（GRPO group sampling） |
| `rollout_id` | `int` | 同一个 rollout 执行产生的多个样本共享此 ID |
| `tokens` | `list[int]` | prompt + response 的完整 token 序列 |
| `loss_mask` | `list[int]` | 1=参与训练，0=不参与（如 tool 调用内容） |
| `rollout_log_probs` | `list[float]` | rollout 引擎产生的 log 概率（用于 off-policy 校正） |
| `rollout_top_p_token_ids` | `list[int]` | top-p 采样的候选 token（用于确定性重放） |
| `rollout_routed_experts` | `list` | MoE routing replay 专家路由信息 |
| `teacher_log_probs` | `list[float]` | 教师模型 log 概率（在线策略蒸馏用） |
| `weight_versions` | `list[str]` | 生成此样本时使用的权重版本（异步训练用） |

### 5.2 其他关键类型

```mermaid
classDiagram
    class RolloutBatch {
        <<type alias: dict>>
        +list~Tensor~ tokens
        +list~Tensor~ loss_masks
        +list~float~ rewards
        +list~int~ response_lengths
        +list~Tensor~ rollout_log_probs
        +list~Tensor~ rollout_top_p_token_ids
    }

    class MultimodalTypes {
        +IMAGE MultimodalType
        +VIDEO MultimodalType
        +AUDIO MultimodalType
        +all() list
        +get(name) MultimodalType
    }

    class MultimodalType {
        +str name
        +str placeholder
    }

    class ParamInfo {
        +str name
        +torch.dtype dtype
        +torch.Size shape
        +dict attrs
        +int size
        +int src_rank
    }

    MultimodalTypes --> MultimodalType
```

---

## 6. 定制化接口

slime 提供 17 个钩子点，覆盖从数据生成到训练的全流程。

### 6.1 接口总览

```mermaid
graph LR
    subgraph DataGen["数据生成阶段"]
        DSP[--data-source-path<br/>数据源]
        RFP[--rollout-function-path<br/>整体 rollout 逻辑]
        CGF[--custom-generate-function-path<br/>单样本生成逻辑]
        CRM[--custom-rm-path<br/>奖励计算]
        EFP[--eval-function-path<br/>评估 rollout]
    end

    subgraph Filtering["过滤阶段"]
        DSF[--dynamic-sampling-filter-path<br/>动态采样过滤]
        BFP[--buffer-filter-path<br/>缓冲区过滤]
        RSFP[--rollout-sample-filter-path<br/>样本级过滤]
        RASPP[--rollout-all-samples-process-path<br/>全样本处理]
    end

    subgraph PostProcess["后处理阶段"]
        RDPP[--rollout-data-postprocess-path<br/>rollout 数据后处理]
        CRPP[--custom-reward-post-process-path<br/>奖励后处理]
        CSDTP[--custom-convert-samples-to-train-data-path<br/>样本转训练数据]
    end

    subgraph Training["训练阶段"]
        CLF[--custom-loss-function-path<br/>自定义 loss]
        CTIS[--custom-tis-function-path<br/>重要性采样]
        CPLR[--custom-pg-loss-reducer-function-path<br/>pg_loss 归约]
        CMIP[--custom-megatron-init-path<br/>Megatron 初始化钩子]
        CMBLP[--custom-megatron-before-log-prob-hook-path<br/>log prob 前钩子]
        CMBTP[--custom-megatron-before-train-step-hook-path<br/>训练步前钩子]
    end

    subgraph Logging["日志阶段"]
        CRLFP[--custom-rollout-log-function-path<br/>训练 rollout 日志]
        CELFP[--custom-eval-rollout-log-function-path<br/>评估 rollout 日志]
    end
```

### 6.2 接口选择决策树

```mermaid
flowchart TD
    Q1{需要什么类型的定制？} --> A1[自定义 Agent 循环/<br/>工具调用/RAG/多轮对话]
    Q1 --> A2[整体替换 rollout 编排]
    Q1 --> A3[自定义奖励计算]
    Q1 --> A4[过滤样本]
    Q1 --> A5[自定义训练目标]
    Q1 --> A6[自定义数据来源]

    A1 --> R1[--custom-generate-function-path<br/>+ --custom-rm-path]
    A2 --> R2[--rollout-function-path<br/>（复杂度高，谨慎使用）]
    A3 --> R3[--custom-rm-path<br/>（支持单样本/批量两种模式）]
    A4 --> Q2{在哪个阶段过滤？}
    Q2 --> R4A[动态采样时<br/>--dynamic-sampling-filter-path]
    Q2 --> R4B[进入训练前<br/>--buffer-filter-path]
    Q2 --> R4C[样本级 loss mask<br/>--rollout-sample-filter-path]
    A5 --> R5[--custom-loss-function-path<br/>+ --loss-type custom_loss]
    A6 --> R6[--data-source-path<br/>继承 DataSource 基类]

    style R1 fill:#4CAF50,color:#fff
    style R2 fill:#FF9800,color:#fff
    style R3 fill:#4CAF50,color:#fff
    style R5 fill:#2196F3,color:#fff
```

### 6.3 关键接口签名

```python
# 1. 自定义生成函数（最常用）
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]

# 2. 自定义奖励函数
async def custom_rm(args, sample: Sample) -> float
# 或批量模式（--group-rm 启用时）
async def batched_custom_rm(args, samples: list[Sample]) -> list[float]

# 3. 整体 rollout 函数
def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput

# 4. 动态采样过滤
def filter_function(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput

# 5. 自定义数据源（继承基类）
class CustomDataSource(DataSource):
    def get_samples(self, num_samples: int) -> list[list[Sample]]: ...
    def add_samples(self, samples: list[list[Sample]]): ...
    def save(self, rollout_id): ...
    def load(self, rollout_id=None): ...
    def __len__(self): ...
```

---

## 7. 部署模式

### 7.1 四种部署架构

```mermaid
graph TB
    subgraph Mode1["模式 1：共址部署（默认）"]
        direction LR
        GPU1[GPU 0..N]
        T1[Training<br/>Megatron] --> GPU1
        R1[Rollout<br/>SGLang] --> GPU1
        Note1["训练和推理共享 GPU<br/>通过内存卸载交替使用"]
    end

    subgraph Mode2["模式 2：分离部署"]
        direction LR
        TGPU[Training GPUs]
        RGPU[Rollout GPUs]
        T2[Training<br/>Megatron] --> TGPU
        R2[Rollout<br/>SGLang] --> RGPU
        T2 -- "delta weight sync" --> R2
        Note2["训练和推理使用独立 GPU<br/>适合超大规模训练"]
    end

    subgraph Mode3["模式 3：外部 Rollout 引擎"]
        direction LR
        T3[Training<br/>Megatron]
        EXT[External SGLang<br/>Server]
        T3 -- "HTTP API" --> EXT
        Note3["SGLang 服务独立管理<br/>可跨不同 GPU 型号/厂商"]
    end

    subgraph Mode4["模式 4：PD 分离"]
        direction LR
        PF[Prefill<br/>SGLang Servers]
        DC[Decode<br/>SGLang Servers]
        RTR[SGLang Router]
        RTR --> PF
        RTR --> DC
        Note4["预填充/解码使用不同资源<br/>适合多轮/agentic 工作负载"]
    end
```

### 7.2 共址 vs 分离部署选择指南

| 场景 | 推荐模式 | 关键参数 |
|------|---------|----------|
| 中小规模模型（7B-70B） | 共址 | 默认配置 |
| 超大规模模型（>100B） | 分离 | `--delta-weight-sync` |
| 推理服务需独立管理 | 外部 | `--external-rollout-engines` |
| 长上下文多轮 agentic | PD 分离 | SGLang Config YAML |
| 混合 GPU 型号 | 外部 + 磁盘传输 | `--update-weight-via-disk` |

---

## 8. 高级特性

### 8.1 权重同步机制

```mermaid
flowchart LR
    subgraph TrainSide["训练侧"]
        TRAIN[Megatron<br/>训练完成]
        PACK[参数打包<br/>ParamInfo]
    end

    subgraph SyncMethods["同步方式"]
        STD_SYNC[标准同步<br/>完整 checkpoint]
        DELTA_SYNC[Delta Weight Sync<br/>增量权重差]
        DISK_SYNC[磁盘传输<br/>共享文件系统]
    end

    subgraph RolloutSide["推理侧"]
        RECV[接收参数]
        UPDATE[更新 SGLang 权重]
        VER[版本记录<br/>weight_versions]
    end

    TRAIN --> PACK
    PACK --> STD_SYNC
    PACK --> DELTA_SYNC
    PACK --> DISK_SYNC
    STD_SYNC --> RECV
    DELTA_SYNC --> RECV
    DISK_SYNC --> RECV
    RECV --> UPDATE
    UPDATE --> VER
```

**Delta Weight Sync** 特别适合：
- 训练/推理分离部署
- 超大模型（减少传输量）
- 不同 GPU 型号之间同步

### 8.2 异步 Rollout（全异步模式）

适用于长尾 agentic 工作负载，部分样本可能需要远长于平均的生成时间。

```mermaid
sequenceDiagram
    participant TL as Training Loop
    participant RM as RolloutManager
    participant BUF as Buffer
    participant SGL as SGLang

    TL->>RM: generate(rollout_id) 请求数据
    RM->>BUF: 从 buffer 取足量数据
    BUF-->>RM: 已有数据（来自上次异步生成）

    par 异步生成下一批
        RM->>SGL: 异步发送生成请求
        SGL-->>RM: 响应（不阻塞训练）
        RM->>BUF: 将完成的样本放入 buffer
    end

    RM-->>TL: 返回当前批次数据
    TL->>TL: 训练当前批次
```

### 8.3 MoE Routing Replay

防止 MoE 模型 RL 训练中的路由不稳定问题。

```mermaid
graph LR
    subgraph Rollout["Rollout 阶段"]
        GEN[生成 token]
        RECORD[记录 expert routing<br/>rollout_routed_experts]
    end

    subgraph Training["训练阶段"]
        FWD[Forward Pass]
        REPLAY1[--use-routing-replay<br/>前向-后向路由一致性]
        REPLAY2[--use-rollout-routing-replay<br/>使用 rollout 时的路由]
    end

    Rollout --> Training
    RECORD --> REPLAY2
```

### 8.4 投机解码（Speculative Decoding）

```mermaid
graph LR
    DRAFT[Draft Model<br/>小模型] --> TOKENS[候选 Token 序列]
    TOKENS --> VERIFY[Target Model<br/>验证]
    VERIFY --> ACCEPT{接受/拒绝}
    ACCEPT -- 接受 --> OUTPUT[输出]
    ACCEPT -- 拒绝 --> RESAMPLE[重采样]

    subgraph Metrics["监控指标"]
        AR[spec_accept_rate<br/>接受率]
        AL[spec_accept_length<br/>平均接受长度]
    end

    OUTPUT --> Metrics
```

### 8.5 观测性体系

```mermaid
graph TB
    subgraph Tracking["训练追踪"]
        WB[WandB<br/>wandb_utils.py]
        TB2[TensorBoard<br/>tensorboard_utils.py]
    end

    subgraph Tracing["轨迹追踪"]
        TU[trace_utils.py<br/>619 行]
        TV[Trace Viewer<br/>可视化工具]
    end

    subgraph Profiling["性能分析"]
        PU[profile_utils.py<br/>GPU profiling]
        FLOPS[flops_utils.py<br/>FLOPS 计算]
        MEMRAY[memray<br/>内存分析]
    end

    subgraph Health["健康监控"]
        HM[health_monitor.py<br/>RolloutHealthMonitor]
        TIMER[timer.py<br/>计时工具]
    end

    subgraph Metrics["指标计算"]
        MU[metric_utils.py<br/>pass_rate/statistics]
        TRAIN_M[train_metric_utils.py]
    end
```

---

## 9. 插件系统

`slime_plugins/` 目录提供可扩展的插件架构：

```mermaid
graph TB
    subgraph Plugins["slime_plugins/"]
        subgraph MBridge["mbridge/ (Megatron Bridge)"]
            MB1[各模型架构适配]
            MB2[权重转换工具]
        end

        subgraph Models["models/"]
            M1[GLM 系列]
            M2[Qwen 系列]
            M3[DeepSeek]
            M4[Llama]
            M5[更多模型...]
        end

        subgraph RolloutBuffer["rollout_buffer/"]
            RB1[缓冲区管理]
            RB2[自定义调度策略]
        end

        subgraph MegatronBridge["megatron_bridge/"]
            BRG1[Megatron ↔ HuggingFace 转换]
            BRG2[权重映射]
        end
    end

    subgraph CoreSlime["核心 slime"]
        CORE[slime/]
    end

    CoreSlime --> Plugins
```

---

## 10. 生态项目

```mermaid
mindmap
  root((slime))
    前沿模型训练
      GLM-5.2 / 5.1 / 5 / 4.7 / 4.6 / 4.5
      Qwen3.6 / 3.5 / 3MoE
      DeepSeek V3 / R1
    框架扩展
      vime
        vLLM 作为 rollout 后端
        vllm-router
      Miles
        企业级特性
        LoRA / TITO 支持
      Relax
        全模态支持
        TransferQueue 异步传输
    Agentic RL
      Dressage
        任意 Agent + 沙盒
        TITO 训练
      OpenClaw-RL
        个性化机器人
        ArenaRL
      qqr / hilichurl
        MCP 集成
        开放式 Agent
      ART
        AWS Bedrock AgentCore
    领域 RL
      P1
        物理奥林匹克
        多阶段 RL
      RLVE
        可验证环境
        400+ 环境联合训练
      TritonForge
        GPU Kernel 生成
        Triton 代码
    系统优化
      APRIL
        主动部分 rollout
        长尾瓶颈优化
```

---

## 11. 配置指南

### 11.1 参数分类体系

```mermaid
graph TB
    subgraph Args["slime 参数体系"]
        MEG_ARGS[Megatron 原生参数<br/>直接传递]
        SGL_ARGS[SGLang 参数<br/>--sglang- 前缀]
        SLIME_ARGS[slime 专有参数<br/>slime/utils/arguments.py]
    end

    subgraph MEGExamples["Megatron 参数示例"]
        ME1[--tensor-model-parallel-size 2]
        ME2[--pipeline-model-parallel-size 4]
        ME3[--micro-batch-size 2]
        ME4[--global-batch-size 128]
    end

    subgraph SGLExamples["SGLang 参数示例"]
        SE1[--sglang-mem-fraction-static 0.8]
        SE2[--sglang-tp-size 2]
        SE3[--sglang-max-running-requests 256]
    end

    subgraph SLIMEExamples["slime 参数示例"]
        SL1[--num-rollout 1000<br/>总 rollout 步数]
        SL2[--group-size 8<br/>每 prompt 生成数]
        SL3[--rollout-function-path<br/>自定义 rollout]
        SL4[--rm-type math<br/>奖励类型]
        SL5[--save-interval 50<br/>保存间隔]
        SL6[--eval-interval 10<br/>评估间隔]
    end

    MEG_ARGS --> MEGExamples
    SGL_ARGS --> SGLExamples
    SLIME_ARGS --> SLIMEExamples
```

### 11.2 核心 slime 参数列表

| 参数类别 | 参数 | 说明 |
|---------|------|------|
| **训练规模** | `--num-rollout` | 总 rollout 步数 |
| | `--group-size` | 每个 prompt 生成的响应数（GRPO） |
| | `--max-response-length` | 最大响应长度 |
| **算法** | `--loss-type` | 损失类型（grpo/ppo/sft/custom_loss） |
| | `--use-critic` | 启用 Critic 模型（PPO） |
| | `--kl-coef` | KL 散度系数 |
| **奖励** | `--rm-type` | 内置奖励类型 |
| | `--custom-rm-path` | 自定义奖励函数路径 |
| | `--rm-url` | 远程奖励服务 URL |
| **数据** | `--train-prompt-data` | 训练 prompt 数据路径 |
| | `--eval-prompt-data` | 评估 prompt 数据路径 |
| | `--data-source-path` | 自定义数据源 |
| **定制化** | `--rollout-function-path` | 自定义 rollout 函数 |
| | `--custom-generate-function-path` | 自定义生成函数 |
| **持久化** | `--save-interval` | checkpoint 保存间隔 |
| | `--eval-interval` | 评估间隔 |
| **内存** | `--offload-train` | 训练时卸载（节省显存） |
| | `--offload-rollout` | rollout 时卸载 |

### 11.3 SGLang Config YAML（高级拓扑配置）

用于需要精细控制的复杂部署（如 PD 分离、多模型服务、异构集群）：

```yaml
# 示例：PD 分离配置
server_groups:
  - name: prefill
    tp_size: 4
    args:
      disaggregation_mode: prefill
      mem_fraction_static: 0.85
  - name: decode
    tp_size: 2
    args:
      disaggregation_mode: decode
      mem_fraction_static: 0.70
router:
  policy: session_affinity  # 多轮对话会话亲和性
```

---

## 12. 测试与 CI

### 12.1 测试层次结构

```mermaid
graph TB
    subgraph TestLevels["测试层次"]
        UNIT[CPU 单元测试<br/>无需 GPU]
        CONTRACT[契约测试<br/>定制化接口]
        E2E[端到端 GPU 测试<br/>完整训练循环]
    end

    subgraph UnitTests["单元测试内容"]
        U1[参数验证<br/>test_megatron_argument_validation.py]
        U2[类型系统测试<br/>Sample/RolloutBatch]
        U3[指标计算<br/>test_metric_report.py]
        U4[Loss 数值一致性<br/>test_loss_cp_invariance.py]
    end

    subgraph ContractTests["契约测试内容"]
        C1[test_plugin_rollout_contracts.py<br/>rollout-function-path]
        C2[test_plugin_generate_contracts.py<br/>custom-generate-function-path]
        C3[test_plugin_path_loading_contracts.py<br/>rm/filter/datasource 接口]
        C4[test_plugin_runtime_hook_contracts.py<br/>运行时钩子接口]
    end

    subgraph E2ETests["E2E 测试内容"]
        E1[Qwen2.5-0.5B 快速测试]
        E2[SGLang Config 测试]
        E3[全异步 Rollout 测试]
        E4[MoE 模型测试]
        E5[多轮对话测试]
        E6[OPD（在线策略蒸馏）测试]
        E7[PPO 测试]
        E8[checkpoint 保存/恢复]
    end

    UNIT --> UnitTests
    CONTRACT --> ContractTests
    E2E --> E2ETests
```

### 12.2 CI 流水线

```mermaid
flowchart LR
    PR[Pull Request] --> PRE[pre-commit<br/>black/isort/ruff]
    PRE --> CPU[CPU 单元测试<br/>run-ci-cpu-unittest]
    CPU --> GPU_LABEL{PR 标签判断}

    GPU_LABEL --> |run-ci-sglang-config| T1[SGLang Config 测试]
    GPU_LABEL --> |run-ci-async| T2[异步 Rollout 测试]
    GPU_LABEL --> |run-ci-disagg| T3[PD 分离测试]
    GPU_LABEL --> |run-ci-moe| T4[MoE 测试]
    GPU_LABEL --> |run-ci-multi-turn| T5[多轮测试]
    GPU_LABEL --> |run-ci-opd| T6[OPD 测试]

    T1 & T2 & T3 & T4 & T5 & T6 --> RESULT[CI 结果]

    style PRE fill:#FF9800,color:#fff
    style CPU fill:#4CAF50,color:#fff
    style RESULT fill:#2196F3,color:#fff
```

### 12.3 pytest 标记

| 标记 | 用途 |
|------|------|
| `unit` | CPU 单元测试 |
| `integration` | 集成测试 |
| `system` | 系统级测试 |
| `acceptance` | 验收测试 |
| `docs` | 文档测试 |
| `skipduringci` | CI 中跳过 |
| `pleasefixme` | 待修复的已知失败 |

---

## 13. 目录结构

```
slime/
├── train.py                    # 同步训练入口
├── train_async.py              # 异步训练入口
├── setup.py                    # 包配置（version 0.3.0）
├── pyproject.toml              # 项目元数据和工具配置
├── requirements.txt            # 依赖列表
│
├── slime/                      # 核心包
│   ├── __init__.py
│   ├── agent/                  # Agent 框架
│   │   ├── aiohttp_threaded.py
│   │   ├── sandbox.py
│   │   ├── trajectory.py
│   │   ├── parsing.py
│   │   ├── adapters/
│   │   └── harness/
│   │
│   ├── rollout/                # Rollout 数据生成
│   │   ├── sglang_rollout.py       # 主 rollout (609 行)
│   │   ├── sglang_streaming_rollout.py
│   │   ├── fully_async_rollout.py  # 全异步 rollout (256 行)
│   │   ├── sft_rollout.py
│   │   ├── on_policy_distillation.py
│   │   ├── data_source.py          # 数据源抽象 (229 行)
│   │   ├── forge_load.py
│   │   ├── base_types.py
│   │   ├── filter_hub/             # 样本过滤
│   │   │   ├── base_types.py
│   │   │   └── dynamic_sampling_filters.py
│   │   └── rm_hub/                 # 奖励模型
│   │       ├── deepscaler.py
│   │       ├── f1.py
│   │       ├── gpqa.py
│   │       ├── ifbench.py
│   │       ├── math_utils.py
│   │       └── math_dapo_utils.py
│   │
│   ├── ray/                    # Ray 分布式编排
│   │   ├── rollout.py              # RolloutManager (1485 行)
│   │   ├── placement_group.py      # GPU 资源分配
│   │   ├── actor_group.py
│   │   ├── train_actor.py
│   │   ├── utils.py
│   │   ├── rollout_validation.py
│   │   └── ray_actor.py
│   │
│   ├── backends/               # 后端实现
│   │   ├── megatron_utils/         # Megatron 工具集 (20+ 子模块)
│   │   │   ├── initialize.py
│   │   │   ├── arguments.py
│   │   │   ├── model.py
│   │   │   ├── actor.py
│   │   │   ├── checkpoint.py
│   │   │   ├── loss.py
│   │   │   ├── data.py
│   │   │   ├── cp_utils.py
│   │   │   ├── kernels/
│   │   │   ├── megatron_patch/
│   │   │   ├── megatron_to_hf/
│   │   │   └── update_weight/
│   │   └── sglang_utils/           # SGLang 工具集
│   │       ├── sglang_engine.py
│   │       ├── arguments.py
│   │       ├── sglang_config.py    # YAML 拓扑配置
│   │       ├── server_control.py
│   │       └── external.py
│   │
│   └── utils/                  # 工具集（32 模块，26K+ 行）
│       ├── arguments.py            # 统一参数解析
│       ├── types.py                # 核心类型（Sample 等）
│       ├── data.py                 # 数据集加载
│       ├── ppo_utils.py            # PPO 算法 (729 行)
│       ├── dp_schedule.py          # 数据并行调度
│       ├── seqlen_balancing.py     # 序列长度均衡
│       ├── mask_utils.py           # Token masking
│       ├── distributed_utils.py    # 多 GPU 协调
│       ├── reloadable_process_group.py
│       ├── async_utils.py
│       ├── http_utils.py           # HTTP 客户端
│       ├── trace_utils.py          # 轨迹追踪 (619 行)
│       ├── logging_utils.py
│       ├── tensorboard_utils.py
│       ├── wandb_utils.py
│       ├── health_monitor.py
│       ├── profile_utils.py
│       ├── metric_utils.py
│       ├── eval_config.py
│       ├── processing_utils.py     # 多模态处理
│       ├── memory_utils.py
│       ├── timer.py
│       ├── misc.py
│       └── ...
│
├── slime_plugins/              # 插件扩展
│   ├── mbridge/                # Megatron Bridge
│   ├── megatron_bridge/
│   ├── models/                 # 模型实现
│   └── rollout_buffer/
│
├── tests/                      # 测试套件 (59 文件)
│   ├── plugin_contracts/       # 接口契约测试
│   ├── test_agent/
│   └── test_*.py
│
├── examples/                   # 示例实现
│   ├── multi_agent/            # 多 agent rollout
│   ├── search-r1/              # 搜索增强多轮生成
│   ├── fully_async/            # 全异步 rollout
│   ├── coding_agent_rl/        # 代码 agent RL
│   ├── geo3k_vlm/              # 视觉语言模型
│   ├── tau-bench/              # TAU benchmark
│   └── ...
│
├── docs/
│   ├── en/                     # 英文文档
│   │   ├── get_started/        # 快速开始
│   │   ├── advanced/           # 高级特性
│   │   ├── developer_guide/    # 开发者指南
│   │   └── blogs/
│   └── zh/                     # 中文文档
│
├── docker/                     # Docker 配置
├── scripts/                    # 辅助脚本
└── tools/                      # 工具集（格式转换/分析）
```

---

## 技术栈总览

```mermaid
graph LR
    subgraph Training["训练"]
        ML[Megatron-LM<br/>分布式训练]
        PT[PyTorch<br/>基础框架]
    end

    subgraph Inference["推理"]
        SG[SGLang<br/>高性能推理]
        SGR[sglang-router<br/>负载均衡]
    end

    subgraph Orchestration["编排"]
        RAY[Ray<br/>分布式系统]
    end

    subgraph Data["数据"]
        HF[HuggingFace<br/>Transformers/Datasets]
        JSONL[JSONL/Parquet]
    end

    subgraph Monitor["监控"]
        WB2[Weights & Biases]
        TB3[TensorBoard]
        MR[memray<br/>内存分析]
    end

    subgraph Quality["代码质量"]
        BLK[black]
        ISO[isort]
        RUF[ruff]
        PRE[pre-commit]
        PT2[pytest]
    end

    Training --- Orchestration
    Inference --- Orchestration
    Orchestration --- Data
    Training --- Monitor
    Inference --- Monitor
```
---

*Wiki 基于 slime v0.3.0 生成 | 更新日期：2026-06-22*