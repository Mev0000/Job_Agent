"""
scripts/distill_7d.py
Track 2 — Gemma 4 离线蒸馏七维属性

功能：
  1. 读取《2022年职业分类大典》CSV（经 load_dictionary.py 预处理为四级节点清单）
  2. 逐条构造 Prompt（跨类Few-Shot + 大典权威声明），调用 Gemma 4 推理七维属性
  3. 质量校验（黑名单过滤 / 完整性检验 / role_level枚举校验）
  4. 增量写入 graph_nodes.json（支持断点续跑）
  5. 生成 Nodes_Cleaned_v2.csv（追加七维列 + category，向后兼容）

使用方式：
  python scripts/distill_7d.py                         # 全量蒸馏 1676 个四级节点（支持断点续跑）
  python scripts/distill_7d.py --dry-run 24            # 跨类验证：1/2/3/4/5/6类各4个
  python scripts/distill_7d.py --resume                # 强制从已有结果断点续跑
  python scripts/distill_7d.py --limit 100             # 只跑前100条（测试用）
  python scripts/distill_7d.py --dict-source data/raw_jd/自定义大典.csv  # 指定大典路径

前置步骤（自动执行）：
  python scripts/load_dictionary.py    # 解析大典CSV → 四级节点清单（自动调用）
  
输入源：data/cache/dict_level4.csv（由 load_dictionary.py 生成，1676 四级节点）
设计依据：融合方案 v5.0 + 大典权威声明 + 跨类Few-Shot（1/2/3/4/6类）
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import re
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_CSV = os.path.join(PROJECT_ROOT, "data", "raw_jd", "2022年职业分类大典（整体修订）.csv")
DICT_L4_CSV = os.path.join(PROJECT_ROOT, "data", "cache", "dict_level4.csv")
# 默认使用 load_dictionary.py 预处理后的四级节点清单
NODES_CSV = DICT_L4_CSV
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "cache", "graph_nodes.json")
OUTPUT_CSV_V2 = os.path.join(PROJECT_ROOT, "data", "graph_tables", "Nodes_Cleaned_v2.csv")
QUALITY_LOG = os.path.join(PROJECT_ROOT, "logs", "distillation_quality.csv")

# 七维字段名
DIMENSIONS = ["core_actions", "objects", "deliverables", "main_kpi", "environment", "served_population", "role_level"]

# ============================================================
# 黑名单体系：复用 clean_meaningless_entities + 按维度分层
# ============================================================
# 策略：
#   core_actions  → ACTION_STOP_WORDS（37动词，严格）— 裸动词无检索区分度
#   objects       → ENTITY_STOP_WORDS（67词，严格）  — 泛化实体稀释匹配精度
#   deliverables  → DELIVERABLE_STOP_WORDS（12词，轻量）— 只拦截纯占位/元数据词
#   environment   → ENTITY_STOP_WORDS（67词，严格）  — 泛化环境词无用
# ============================================================

# 确保 scripts/ 在 path 中，以便 import clean_meaningless_entities
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from clean_meaningless_entities import ACTION_STOP_WORDS, ENTITY_STOP_WORDS, GARBAGE_PREFIXES, GARBAGE_SUFFIXES, is_meaningless
    print("✅ 已加载专家级黑名单：clean_meaningless_entities（37动词 + 67实体/环境泛词）")
except ImportError:
    ACTION_STOP_WORDS = {"负责", "参与", "协助", "处理", "完成", "进行", "开展", "做好", "配合",
                          "从事", "提供", "实施", "组织", "办理", "服务", "使用", "执行", "利用",
                          "制定", "确定", "提出", "担任", "建立", "协调", "履行", "沟通"}
    ENTITY_STOP_WORDS = {"工作", "业务", "设备", "系统", "文档", "资料", "原料", "物料", "项目",
                          "任务", "人员", "问题", "活动", "机构", "单位", "办公室", "生产线"}
    GARBAGE_PREFIXES = r'^(相关|其他|各种|各类|有关|某些|一般|日常的?|简单的?)'
    GARBAGE_SUFFIXES = r'(等工作|等相关工作|及其他|等事项|等相关事宜|等)$'
    def is_meaningless(entity_str, stop_words_set):
        if not entity_str: return True
        return str(entity_str).strip() in stop_words_set
    print("⚠️ 使用兜底黑名单（建议在 scripts/ 目录下执行以加载完整黑名单）")


# === objects 专用黑名单 ===
# 从 ENTITY_STOP_WORDS 继承 67 词（实体/环境泛词），额外追加管理类泛词
# 这些是指"单独出现时过于泛化"的词，「修饰词+核心词」的合法用法不会被误杀
OBJECT_BLACKLIST = set(ENTITY_STOP_WORDS) | {
    # 管理类泛词（LLM 容易输出，但对检索无区分度）
    "工作机构", "组织资源", "机构人员", "日常政务", "行政事务",
    "有关工作", "相关工作", "各项工作", "业务工作", "各项工作内容",
    # 空洞名词（任何职业都可能出现）
    "行政工作", "管理工作", "组织工作", "党务工作", "日常事务",
}

# === objects 专用白名单（精确豁免集合）===
# 这些词在 ENTITY_STOP_WORDS 里（因为来自旧 JD NER 时确实泛化），
# 但在《职业分类大典》任务描述语境下是合法的专业具体词，不应被黑名单误杀。
# 判断依据：大典官方任务描述中出现该词时，代表该职业的核心工作对象（如操作工艺装备、记录工艺参数）
OBJECT_WHITELIST = {
    # 制造业技术参数/数据词（6类高频，如冶金/化工/纺织等操作岗）
    "工艺参数", "运行参数", "生产数据", "生产记录",
    # 制造业设备词（ENTITY_STOP_WORDS 里有裸词"设备"，但这些是有修饰的合法复合词）
    "工艺装备", "加工设备", "仪器仪表", "成型设备", "附属设备",
    "消防救生设备", "环保设施", "生产设备", "配料设备",
    # 原料/物料词（在生产加工岗位是明确工作对象，如原材料入库、原辅料配比）
    "原材料", "原辅料", "原料和产品",
    # 生产场所/设施词（作为 objects 时指"作用对象"，如维护生产线、检查集材道）
    "生产线", "集材道",
    # 其他专业词
    "各类资料",   # 3-01-02-04 档案管理相关，资料本身就是工作对象
}

# === 交付物专用轻量黑名单 ===
# 只在"绝无可能是交付物"时才拦截，比 objects 宽松很多。
# 「工作报告」「工作计划」「会议纪要」等在 deliverables 语境下都是合法输出，
# 且由于三级漏斗的精确匹配特性，"会议纪要" ≠ "会议"，不会误杀。
DELIVERABLE_STOP_WORDS = {
    # 纯占位/元数据词（绝无可能是交付物）
    "相关", "其他", "等工作", "及其他", "及其他事项", "等有关",
    "有关规定", "其他职责", "等情况", "领导职务", "职权",
    "职责", "负责",
}


# ============================================================
# 1. Prompt 构造
# ============================================================

DISTILL_PROMPT = """你是国家职业分类大典的语义分析专家。请仔细阅读以下职业定义，提取该职业的七个核心维度。

⚠️ **重要声明**：这是《中华人民共和国职业分类大典（2022年版）》的官方条目。职业名称和描述100%权威，不存在头衔拔高或名实不符的问题。你只需**忠实提取**，无需怀疑数据真实性。

## 职业信息
- 职业编码：{code}
- 职业名称：{name}
- 职业描述：{desc}
- 主要工作任务：{tasks}

## 输入使用策略
- 若「主要工作任务」非空 → 以它为**主要依据**（大典官方编写的任务清单，最准确）
- 若「主要工作任务」为空 → 以「职业描述」为主要依据

---

## Few-Shot 示例（跨5个大类，展示不同责任层级的判定法则）

### 示例1：1类·管理决策 → 全面经营负责
职业编码：1-02-02 | 职业名称：国家行政机关负责人
职业描述：在各级人民政府及其工作部门中，担任领导职务并具有决策、管理职权的人员。
主要工作任务：（空）
```json
{{
  "core_actions": ["行政决策", "资源配置", "人事任免", "政策制定"],
  "objects": ["政府机关", "行政资源", "行政人员", "公共政策"],
  "deliverables": ["年度工作报告", "政策文件", "人事任免决定"],
  "main_kpi": "政策落实率",
  "environment": ["政府办公大楼", "政务大厅", "基层现场"],
  "served_population": ["社会公众", "企业单位", "上级政府"],
  "role_level": "全面经营负责"
}}
```
▶ 判定理由：JD含"决策/管理职权/资源配置"等决策性动词 → 全面经营负责

### 示例2：2类·专业技术 → 专业/项目负责
职业编码：2-01-02-00 | 职业名称：经济学研究人员
职业描述：从事经济学理论研究，运用经济学原理和经济规律对经济问题提出解决办法的专业人员。
主要工作任务：1.研究商品和劳务的生产、分配、交换和消费及其衍生的市场交易趋势...；2.收集和分析经济资料，建立数学模型，说明和预测经济行为...；3.研究经济制度、经济发展史、经济思想史和经济学方法论；4.研究产业、区域和财政、税收、金融、国际贸易等领域经济问题...
```json
{{
  "core_actions": ["研究经济理论", "分析经济数据", "建立数学模型", "撰写研究报告"],
  "objects": ["经济数据", "经济制度", "市场趋势"],
  "deliverables": ["研究报告", "经济预测报告", "政策建议"],
  "main_kpi": "研究成果质量",
  "environment": ["研究机构", "高等院校", "数据中心"],
  "served_population": ["政府部门", "企业", "学术界"],
  "role_level": "专业/项目负责"
}}
```
▶ 判定理由：JD含"研究/分析/建立模型/提出解决办法"等创制性动词 → 专业/项目负责

### 示例3：3类·办事辅助 → 事务支持
职业编码：3-01-01-01 | 职业名称：行政办事员
职业描述：在公共管理和社会组织机构中，从事具体行政业务办理，以及基层人民政府和派出机构中，从事司法助理、民政助理等行政业务的人员。
主要工作任务：1.草拟规章、政策或实施细则等文件；2.了解规章、政策执行情况，指导有关工作，报告调查研究情况；3.提出有关业务工作改进建议；4.接待来访者和办事人员，处理有关事宜。
```json
{{
  "core_actions": ["草拟规章文件", "调研政策执行", "接待来访人员", "办理行政业务"],
  "objects": ["规章政策", "来访人员", "业务材料"],
  "deliverables": ["规章草案", "调研报告", "业务办理回执"],
  "main_kpi": "业务办理时效",
  "environment": ["行政办公区", "政务服务大厅"],
  "served_population": ["办事群众", "企事业单位"],
  "role_level": "事务支持"
}}
```
▶ 判定理由：JD含"草拟/接待/办理/协助"等辅助性动词，不承担管理责任 → 事务支持

### 示例4：4类·社会生产生活服务 → 一线执行
职业编码：4-01-02-01 | 职业名称：营销员
职业描述：从事市场调查、市场分析、营销策划、商品与服务推广等工作的人员。
主要工作任务：1.调查了解市场信息，分析、预测、开发市场，寻找潜在客户；2.运用市场分析方法，制订营销策划方案，推广产品、提供服务、洽谈合作；3.提供售前、售中、售后服务；4.办理商品的交付、发运；5.处理商品销售过程中的纠纷；6.签订销售和服务合同；7.结算货款；8.维护客户关系。
```json
{{
  "core_actions": ["调查市场信息", "制订营销方案", "推广产品服务", "维护客户关系"],
  "objects": ["市场信息", "产品", "客户"],
  "deliverables": ["营销方案", "销售合同", "客户档案"],
  "main_kpi": "销售额",
  "environment": ["线下门店", "客户公司", "市场区域"],
  "served_population": ["消费者", "企业客户"],
  "role_level": "一线执行"
}}
```
▶ 判定理由：JD含"调查/推广/服务/签订"等直接执行动词，不承担管理责任 → 一线执行

### 示例5：6类·生产制造 → 一线执行
职业编码：6-01-01-01 | 职业名称：制米工
职业描述：操作粮食加工机械，将原粮制成成品米的生产人员。
主要工作任务：1.操作清理、计量、输送等设备，接收、清理原粮；2.操作砻谷机、谷糙分离等设备，脱壳、分离谷糙；3.操作碾米、抛光、色选、配米等设备，精加工糙米；4.操作称重、打包、封口设备，计量、包装成品；5.处理、回收生产过程中的副产品及下脚料...
```json
{{
  "core_actions": ["操作清理设备", "脱壳分离谷糙", "碾米抛光", "包装成品"],
  "objects": ["原粮", "加工设备", "成品米"],
  "deliverables": ["成品米", "生产记录"],
  "main_kpi": "成品合格率",
  "environment": ["粮食加工车间", "仓储区"],
  "served_population": ["下游加工企业", "消费者"],
  "role_level": "一线执行"
}}
```
▶ 判定理由：JD含"操作/加工/包装"等执行性动词 → 一线执行

---

## ⚠️ 核心规则：禁止裸泛词，必须写具体词组

以下词如果**单独出现**（没有修饰语）会被质量校验拦截，必须写成「修饰词+核心词」的具体形式：

### 禁止的裸动词（必须带宾语/状语）
❌ 协调 → ✅ 协调劳动关系 / 协调部门合作
❌ 组织 → ✅ 组织选举 / 组织培训活动
❌ 制定 → ✅ 制定发展战略 / 制定管理规范
❌ 开展 → ✅ 开展调查研究 / 开展民主监督
❌ 负责 → ✅ （不用这个词，直接用具体动作）
❌ 参与/协助/处理/进行/做好 → ✅ （不用这些词，用具体动作）
❌ 实施/执行/运用/办理/提供 → ✅ 必须有具体宾语

### 禁止的裸实体（必须具体化）
❌ 机构 → ✅ 政府机关 / 司法机关 / 监察机构
❌ 单位 → ✅ 企业单位 / 事业单位 / 基层单位
❌ 设备 → ✅ （用具体设备名）
❌ 系统 → ✅ （用具体系统名，如"审判系统"）
❌ 人员 → ✅ （用具体角色，如"审判人员""执法人员"）

### deliverables（交付物）要求
- "工作报告""工作计划""会议纪要" = ✅ 通过（在政府/管理语境中是合法交付物）
- "相关工作""其他""等" = ❌ 拦截
- deliverables 不能为空，至少要提取 1 个具体交付物

### objects 具体性要求
- ⚠️ objects 必须极度具体！禁止用任何泛化词：
  - ❌ 「工作机构」→ ✅ 「人大常委会工作机构」
  - ❌ 「组织资源」→ ✅ （不用这个词，用具体资源名，如「审判法庭」「监察装备」）
  - ❌ 「机构人员」→ ✅ 「公职人员」「监察人员」「法律文书人员」
  - ❌ 「行政事务」→ ✅ 「行政规章制度」「行政法律事务」

### deliverables 区分度要求
- ⚠️ deliverables 必须有区分度！
  - 如果多个职业都有「年度工作报告」，必须加修饰语区分：
    - ✅ 「人大常委会年度工作报告」（权力机关）
    - ✅ 「人民法院年度工作报告」（审判机关）
  - 优先提取更具体的交付物（如「判决书」「监察决定书」> 「年度工作报告」）

### environment 具体性要求
- ⚠️ environment 必须具体化！不能写通用办公场所：
  - ❌ 「办公大楼」→ ✅ 「人民法院办公区」/「监察机关办案区」
  - ❌ 「会议中心」→ ✅ 「人大常委会会议中心」/「审判委员会会议室」
  - ❌ 「政府办公大楼」→ ✅ 加所属单位前缀

---

## 提取要求

请按以下规则提取，输出严格 JSON：

1. **core_actions**（核心动作，list[str]，2-5个）：该职业最**本质、最具区分度**的动词词组。从「主要工作任务」中提炼每个任务的**核心动作**。
   - 参考示例中的提取方式：从"1.草拟规章..." → "草拟规章文件"；从"1.操作清理设备..." → "操作清理设备"
   - ❌ 坏例：["负责管理", "协调工作", "处理事务"] — 毫无区分度

2. **objects**（作用对象，list[str]，2-4个）：动作直接施加的具体实体。不要填"工作""业务""事项"等万能词。

3. **deliverables**（交付物，list[str]，1-3个）：该职业最终输出的**可见、可交付**的成果。这是最容易遗漏但最重要的维度。
   - ❌ 坏例：[]（空列表直接拒绝）；["相关工作"]（占位词）

4. **main_kpi**（核心考核指标，str）：量化该职业成败的关键指标。写最能体现价值的那一个。
   - 如果确实无法从文本中提取 → 写 "无明确指标"

5. **environment**（工作环境，list[str]，1-3个）：物理/数字空间。必须**具体化**。

6. **served_population**（服务对象，list[str]，1-3个）：该职业的服务受益者。

7. **role_level**（责任层级，str）：该岗位的**责任大小**，不是职业大类！职业大类（category）由编码第1位解析，不需要你判断。必须从以下值选1：
   - `"全面经营负责"` = 对组织/机构/经营单元的结果承担**最终、全面**责任。判断依据：JD含"战略制定/资源配置/组织绩效终审"等决策性动词
   - `"专业/项目负责"` = 对某个专业领域或项目负总责，但**不对整体经营**负责。判断依据：JD含"从0到1研发/技术方案设计/项目交付"等创制性动词
   - `"团队管理"` = 管理一个团队，对团队产出负责，但不是最终决策者。判断依据：JD含"团队排班/绩效考核/日常管理"等管理动词
   - `"一线执行"` = 直接执行具体工作，不负有管理责任。判断依据：JD不含管理责任，直接执行具体工作
   - `"事务支持"` = 提供辅助性、支持性工作。判断依据：JD含"文书处理/档案管理/前台接待"等辅助动词

   ⚠️ **注意**：头衔中的"经理/总监/专家/负责人"不等于role_level，必须以【核心动作动词】判定！参考示例：行政办事员虽然有"办事"二字，但核心动作是"草拟/接待/办理" → 事务支持，而不是"一线执行"。

## 输出格式（严格 JSON，不要额外解释）

```json
{{
  "core_actions": [...],
  "objects": [...],
  "deliverables": [...],
  "main_kpi": "...",
  "environment": [...],
  "served_population": [...],
  "role_level": "..."
}}
```

## 最后提醒
- **宁可空着也不要瞎编**！如果某项确实无法从文本提取，填 `[]`（列表）或 `"无明确指标"`（KPI）
- **不要裸词**！每个动词和实体都要加修饰语，否则会被质量校验拒绝
- **role_level 判定看JD动词**：不是看名称头衔，而是看核心动作（参考示例1-5的判定理由）
- **用「主要工作任务」为主要依据**：大典的「主要工作任务」是官方编写的任务清单，比「职业描述」更具体
"""


# ============================================================
# 2. 质量校验
# ============================================================

def _get_category(code: str) -> str:
    """
    从职业编码前缀直接判定职业大类（不需要 LLM 推理）。
    编码规则：第1位 = 大类
      1 → 管理决策    2 → 专业技术    3 → 办事辅助
      4 → 生产生活服务  5 → 农林牧渔    6 → 生产制造    7 → 军队
    """
    first = code.split("-")[0].strip()
    mapping = {
        "1": "1类-管理决策",
        "2": "2类-专业技术创制",
        "3": "3类-办事辅助",
        "4": "4类-生产生活服务",
        "5": "5类-农林牧渔",
        "6": "6类-生产制造",
        "7": "7类-军队",
    }
    return mapping.get(first, "8类-不便分类")


def validate_7d(result_json: dict, code: str, name: str) -> tuple[bool, str]:
    """
    质量校验：完整性 + 分层黑名单（不同维度用不同黑名单）。
    
    分层逻辑（v2.0）：
      core_actions  → ACTION_STOP_WORDS（37动词，严格）— 裸动词无检索区分度，必须拒绝
      objects       → ENTITY_STOP_WORDS（67词，严格）— 泛化实体稀释检索精度
      deliverables  → DELIVERABLE_STOP_WORDS（12词，轻量）— 只拦截纯占位/元数据词
      environment   → ENTITY_STOP_WORDS（67词，严格）— 泛化环境词无用
    
    返回：(is_valid, reject_reason)
    """
    # === 完整性检验 ===
    for dim in ["core_actions", "deliverables"]:
        val = result_json.get(dim, None)
        if not val or (isinstance(val, list) and len(val) == 0):
            return False, f"{dim} 为空（完整性检验未通过）"
    
    # === role_level 枚举校验（责任层级，不是职业大类）===
    valid_levels = [
        "全面经营负责", "专业/项目负责", "团队管理",
        "一线执行", "事务支持",
    ]
    rl = result_json.get("role_level", "")
    if rl and rl not in valid_levels:
        return False, f"role_level='{rl}' 不在枚举范围内（应为责任层级：{valid_levels}）"
    
    # === core_actions — ACTION_STOP_WORDS 严格校验 ===
    for a in result_json.get("core_actions", []):
        if is_meaningless(a, ACTION_STOP_WORDS):
            return False, f"core_actions 含通用动词「{a}」（37动词黑名单：裸词无检索区分度，请用具体词组如「协调劳动关系」代替「协调」）"
    
    # === objects — OBJECT_BLACKLIST 严格校验（inherit ENTITY_STOP_WORDS + 管理类泛词） ===
    # 注意：先检查白名单豁免，再检查黑名单。
    # OBJECT_WHITELIST 收录了"看起来泛化但在大典任务语境下是合法专业词"的词，如"工艺参数""工艺装备"
    for o in result_json.get("objects", []):
        if o in OBJECT_WHITELIST:
            continue  # 白名单豁免，跳过黑名单校验
        if is_meaningless(o, OBJECT_BLACKLIST):
            return False, f"objects 含泛化实体「{o}」（黑名单拦截：过于泛化无区分度，请用具体对象）"
    
    # === deliverables — DELIVERABLE_STOP_WORDS 轻量校验 ===
    # 注意：仅拦截纯占位/元数据词（如"相关""其他""等工作"），不拦截"工作报告""工作计划"等
    # 因为这些在 deliverables 语境下是合法输出，且三级漏斗不会误杀复合词（"会议纪要"≠"会议"）
    for d in result_json.get("deliverables", []):
        if is_meaningless(d, DELIVERABLE_STOP_WORDS):
            return False, f"deliverables 含纯占位词「{d}」（轻量黑名单：仅拦截元数据/占位词）"
    
    # === environment — ENTITY_STOP_WORDS 严格校验 ===
    for e in result_json.get("environment", []):
        if is_meaningless(e, ENTITY_STOP_WORDS):
            return False, f"environment 含泛化环境词「{e}」（67词黑名单：请用具体场所如「政务大厅」代替「办公室」）"
    
    return True, ""


# ============================================================
# 3. LLM 调用（复用 Gemma4Client 或直调 Ollama）
# ============================================================

def get_llm_client(config_path: str = None):
    """获取 LLM 客户端（优先用项目内置 Gemma4Client）"""
    import yaml
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 尝试导入项目内的 Gemma4Client
    try:
        import sys
        sys.path.insert(0, PROJECT_ROOT)
        from core.llm_client import Gemma4Client
        client = Gemma4Client(config)
        print("✅ 使用项目内置 Gemma4Client")
        return client, config
    except Exception as e:
        print(f"⚠️ 无法导入 Gemma4Client：{e}")
        print("   将使用直调 Ollama API 的方式...")
        return None, config


def call_distill(llm_client, prompt: str) -> dict:
    """
    调用 LLM 推理七维属性。
    返回解析后的 JSON dict，解析失败返回 {}
    """
    try:
        # Gemma4Client 接口：generate(system_prompt, user_prompt, ...) -> str
        raw_response = llm_client.generate(
            system_prompt="你是一个国家职业分类语义分析专家，严格遵循用户指令输出JSON。",
            user_prompt=prompt,
            temperature=0.1,   # 蒸馏任务需要确定性输出
            max_tokens=4096,   # 七维输出需要较大 token 空间
        )
        if not raw_response or not raw_response.strip():
            print(f"    ⚠️ LLM 返回空响应")
            return {}

        # === 多层 JSON 提取策略 ===
        json_str = None

        # 策略1：剥去 ```json ... ``` 包裹（包括截断缺少闭合 ``` 的情况）
        m = re.search(r'```json\s*([\s\S]*?)(?:\s*```|$)', raw_response)
        if m:
            json_str = m.group(1).strip()
        else:
            # 策略2：直接找最外层 { ... }
            m = re.search(r'(\{[\s\S]*\})', raw_response)
            if m:
                json_str = m.group(1).strip()

        if not json_str:
            print(f"    ⚠️ 未找到 JSON 结构，原始输出：{raw_response[:200]}")
            return {}

        # 策略3：标准解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 策略4：截断修复 — 补齐缺失的 ] } 和引号
        repaired = _repair_truncated_json(json_str)
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

        print(f"    ⚠️ JSON 解析失败，原始输出（前200字）：{raw_response[:200]}")
        return {}
    except Exception as e:
        print(f"    ❌ LLM 调用失败：{e}")
        return {}


def _repair_truncated_json(truncated: str) -> str:
    """
    尝试修复被 max_tokens 截断的 JSON：
    - 补齐缺失的 ]
    - 补齐缺失的 }
    - 补齐缺失的尾引号
    """
    s = truncated.rstrip()
    # 补齐数组中缺失的 ]
    open_brackets = s.count('[') - s.count(']')
    s += ']' * max(0, open_brackets)
    # 补齐对象中缺失的 }
    open_braces = s.count('{') - s.count('}')
    s += '}' * max(0, open_braces)
    # 补齐尾引号（简单启发式：最后非空白字符在引号内）
    s = s.rstrip()
    if s.endswith(',') or s.endswith(':') or s.endswith('['):
        # 值被截断，用 null 占位
        s += 'null'
    # 补齐缺失的尾引号（如果引号不成对）
    in_string = False
    for ch in s:
        if ch == '"':
            in_string = not in_string
    if in_string:
        s += '"'
    return s


# ============================================================
# 4. 断点续跑：加载已有结果
# ============================================================

def load_existing_results(output_json: str) -> dict:
    """加载已有蒸馏结果（支持断点续跑）"""
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results_incremental(output_json: str, node_id: str, node_data: dict):
    """增量保存（每次写入整文件，JSON 格式支持人工编辑）"""
    data = load_existing_results(output_json)
    data[node_id] = node_data
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# 5. 主流程
# ============================================================

def run_distillation(
    dry_run: int = 0,
    limit: int = 0,
    resume: bool = True,
    quality_strategy: str = "delete",  # delete / fallback / review / skip
    rerun_low: bool = False,  # True = 重跑所有 _quality=="LOW" 的节点（白名单修复后补跑）
):
    """
    quality_strategy:
      - delete（默认）：低质量结果不保存，从蒸馏结果中删除（纯净化图谱）
      - fallback：低质量节点保留旧三维属性，不强制更新（但 core_actions 仍写入）
      - review：标记为 needs_human_review
      - skip：只更新高质量节点
    """
    print("=" * 60)
    print("🔬 Gemma 4 离线蒸馏七维属性 — 启动")
    print("=" * 60)
    
    # 加载数据：优先使用 dict_level4.csv，不存在则自动从大典CSV生成
    if not os.path.exists(NODES_CSV):
        if os.path.exists(DICT_CSV):
            print(f"⚠️ dict_level4.csv 不存在，自动从大典CSV生成...")
            import subprocess
            subprocess.run([
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "load_dictionary.py"),
                "--csv", DICT_CSV,
                "--output", DICT_L4_CSV,
            ], check=True)
        else:
            print(f"❌ 找不到输入源：\n   {NODES_CSV}\n   {DICT_CSV}")
            return
    if not os.path.exists(NODES_CSV):
        print(f"❌ 找不到四级节点文件：{NODES_CSV}")
        print(f"   请先运行 python scripts/load_dictionary.py 生成")
        return
    df = pd.read_csv(NODES_CSV).fillna("")
    print(f"✅ 已加载 {len(df)} 条职业节点")
    
    # === dry-run 跨类均匀采样 ===
    if dry_run > 0:
        # 为每条记录计算 category（从编码第1位解析）
        def _cat_code(code_str):
            first = str(code_str).split("-")[0].strip()
            return int(first) if first.isdigit() and first != "7" else 0
        
        df["_cat_code"] = df["职业编码"].apply(_cat_code)
        
        # 目标类别 1-6（排除7类军队），每类均匀采样
        target_cats = [1, 2, 3, 4, 5, 6]
        per_cat = max(1, dry_run // len(target_cats))
        remaining = dry_run - per_cat * len(target_cats)
        
        sampled_rows = []
        for i, cat in enumerate(target_cats):
            cat_df = df[df["_cat_code"] == cat]
            # 前 remaining 个类多取 1 条以凑齐 dry_run 总数
            n_sample = min(per_cat + (1 if i < remaining else 0), len(cat_df))
            if n_sample > 0:
                sampled = cat_df.sample(n=n_sample, random_state=42)
                sampled_rows.append(sampled)
                print(f"   📊 第{cat}类：采样 {n_sample}/{len(cat_df)} 条")
            else:
                print(f"   ⚠️ 第{cat}类：无可用节点（跳过）")
        
        if sampled_rows:
            df = pd.concat(sampled_rows, ignore_index=True)
        else:
            print("❌ 所有目标类别均无可用节点，终止")
            return
        df = df.drop(columns=["_cat_code"])
        print(f"✅ dry-run 跨类均匀采样完成：共 {len(df)} 条\n")
    
    # 断点续跑：加载已有结果
    existing = load_existing_results(OUTPUT_JSON) if resume else {}
    if existing:
        if rerun_low:
            # --rerun-low 模式：把 LOW 质量的节点从 existing 中移除，强制重跑
            low_keys = [k for k, v in existing.items() if v.get("_quality") == "LOW"]
            for k in low_keys:
                del existing[k]
            print(f"🔄 断点续跑：已有 {len(existing) + len(low_keys)} 条（其中 {len(low_keys)} 条 LOW 质量将重跑）")
        else:
            print(f"🔄 断点续跑：已有 {len(existing)} 条结果，将跳过")
    
    # 获取 LLM 客户端
    llm_client, config = get_llm_client()
    if llm_client is None:
        print("❌ 无法初始化 LLM 客户端，中止")
        return
    
    os.makedirs(os.path.dirname(QUALITY_LOG), exist_ok=True)
    quality_rows = []
    
    processed = 0
    skipped = 0
    failed = 0
    low_quality = 0
    
    for idx, row in df.iterrows():
        code = str(row.get("职业编码", "")).strip()
        name = str(row.get("职业名称", "")).strip()
        desc = str(row.get("职业描述", "")).strip()
        tasks = str(row.get("主要工作任务", "")).strip()
        
        if not code or not name:
            skipped += 1
            continue
        
        # 断点续跑跳过
        if code in existing:
            skipped += 1
            continue
        
        # dry-run 模式
        if dry_run > 0 and processed >= dry_run:
            print(f"\n🛑 dry-run 达到 {dry_run} 条，停止")
            break
        if limit > 0 and processed >= limit:
            print(f"\n🛑 达到 --limit {limit} 条，停止")
            break
        
        # 构造 Prompt
        prompt = DISTILL_PROMPT.format(
            code=code,
            name=name,
            desc=desc[:500],   # 截断，防止上下文溢出
            tasks=tasks[:800],
        )
        
        print(f"\n▶️ [{processed+1}/{len(df)}] 蒸馏：{code} {name}")
        
        # 调用 LLM
        t0 = time.time()
        result_json = call_distill(llm_client, prompt)
        elapsed = time.time() - t0
        print(f"   ⏱️ 推理耗时：{elapsed:.1f}s")
        
        if not result_json:
            failed += 1
            quality_rows.append({
                "code": code, "name": name,
                "status": "FAILED", "reason": "JSON解析失败",
                "timestamp": datetime.now().isoformat(),
            })
            continue
        
        # 质量校验
        is_valid, reason = validate_7d(result_json, code, name)
        
        if not is_valid:
            # ⚠️ 低质量结果不拦截，标记后继续存入（方便人工/后续审核）
            low_quality += 1
            print(f"   ⚠️ 质量校验未通过（已标记）：{reason}")
            quality_rows.append({
                "code": code, "name": name,
                "status": "LOW_QUALITY", "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })
            node_data = {
                **result_json,
                "node_id": code, "name": name,
                "category": _get_category(code),
                "_quality": "LOW",
                "_reject_reason": reason,
            }
        else:
            print(f"   ✅ 质量校验通过")
            node_data = {**result_json, "node_id": code, "name": name, "category": _get_category(code), "_quality": "OK"}
        
        # 增量保存
        save_results_incremental(OUTPUT_JSON, code, node_data)
        processed += 1
        
        # 每10条打印一次进度
        if processed % 10 == 0:
            print(f"   📊 进度：已处理 {processed} 条 | 跳过 {skipped} 条 | 失败 {failed} 条 | 低质量 {low_quality} 条")
    
    # 保存质量日志
    if quality_rows:
        pd.DataFrame(quality_rows).to_csv(QUALITY_LOG, index=False, encoding="utf-8-sig")
        print(f"\n📋 质量日志已保存：{QUALITY_LOG}")
    
    # 生成 Nodes_Cleaned_v2.csv（含七维属性 + category）
    print(f"\n📊 正在生成 {OUTPUT_CSV_V2} ...")
    final_data = load_existing_results(OUTPUT_JSON)
    df_v2 = df.copy()
    for dim in DIMENSIONS:
        df_v2[dim] = df_v2["职业编码"].apply(
            lambda c: json.dumps(final_data.get(str(c).strip(), {}).get(dim, []), ensure_ascii=False)
            if dim != "role_level" else final_data.get(str(c).strip(), {}).get(dim, "")
        )
    # 附加 category（职业大类，从编码解析，非 LLM 蒸馏）
    df_v2["category"] = df_v2["职业编码"].apply(
        lambda c: final_data.get(str(c).strip(), {}).get("category", _get_category(str(c)))
    )
    df_v2.to_csv(OUTPUT_CSV_V2, index=False, encoding="utf-8-sig")
    print(f"✅ Nodes_Cleaned_v2.csv 已生成（含七维属性 + category职业大类），共 {len(df_v2)} 条")
    
    # 统计
    print("\n" + "=" * 60)
    print("📊 蒸馏完成统计：")
    print(f"   ✅ 成功处理：{processed} 条")
    print(f"   ⏭️ 断点跳过：{skipped} 条")
    print(f"   ❌ 推理失败：{failed} 条")
    print(f"   ⚠️ 低质量：{low_quality} 条（策略：{quality_strategy}）")
    print(f"   📂 结果文件：{OUTPUT_JSON}")
    print("=" * 60)


# ============================================================
# 6. CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 离线蒸馏七维属性")
    parser.add_argument("--dry-run", type=int, default=0, help="dry-run 模式，跨类均匀采样 N 条（1-6类各均分）")
    parser.add_argument("--limit", type=int, default=0, help="限制处理条数（测试用）")
    parser.add_argument("--no-resume", action="store_true", help="不从断点续跑（重新从头开始）")
    parser.add_argument("--dict-source", type=str, default=None,
                        help="大典CSV路径（默认: data/raw_jd/2022年职业分类大典（整体修订）.csv）")
    parser.add_argument("--rerun-low", action="store_true",
                        help="重跑所有 _quality==LOW 的节点（白名单修复后补跑，其余节点正常断点续跑）")
    parser.add_argument("--quality-strategy", type=str, default="delete",
                        choices=["delete", "fallback", "review", "skip"],
                        help="低质量结果处理策略（fallback=保留旧属性/review=标记人工/skip=跳过）")
    args = parser.parse_args()
    
    # 支持自定义大典路径
    if args.dict_source:
        DICT_CSV = os.path.join(PROJECT_ROOT, args.dict_source) if not os.path.isabs(args.dict_source) else args.dict_source
        # 自定义路径时重新生成 L4
        DICT_L4_CSV = os.path.join(os.path.dirname(DICT_CSV), "dict_level4.csv")
        NODES_CSV = DICT_L4_CSV
    
    run_distillation(
        dry_run=args.dry_run,
        limit=args.limit,
        resume=not args.no_resume,
        quality_strategy=args.quality_strategy,
        rerun_low=args.rerun_low,
    )
