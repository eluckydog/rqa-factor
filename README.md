# RQA Factor · 递归量化因子

> Recurrence plots from phase-space embeddings capture nonlinear regime transitions via determinism (DET) and laminarity (LAM).

基于相空间递归图（Recurrence Plot）的非线性市场状态检测因子。

## ⚠️ 手感练习声明

本项目仅为**挖掘量化因子的手感练习**，未经实测验证。

- 合成数据测试通过，不代表实盘有效
- 未接入实际行情做回测
- RQA 参数（嵌入维度、延迟、阈值）为启发式默认值，未优化
- 不构成投资建议，使用后果自负

当你看到这个因子时，请理解：我们还在练习

## 核心思想

将价格收益率序列嵌入到相空间，构建递归图（Recurrence Plot），从递归结构提取 6 个非线性动力学指标。

```
收益率序列 → 时滞嵌入（Takens 定理） → 相空间向量
    ↓
递归矩阵（阈值化距离）
    ↓
RQA 指标：RR, DET, Lmax, ENTR, LAM, TT
```

**信号逻辑**：市场状态变化会改变收益率时间序列的递归结构。

- **DET（确定性）上升** → 市场变得更加规则、可预测（危机模式）
- **Lmax（最长对角线）延长** → 递归结构中周期成分增长
- **ENTR（熵）下降** → 对角线长度分布集中化
- **LAM（层流率）上升** → 系统陷入"陷阱态"，趋势持续

## 可观测量

| 指标 | 含义 | 危机信号 |
|------|------|---------|
| **RR** | 递归率：相空间点之间的距离 < 阈值 | 固定~10%（由阈值控制） |
| **DET** | 确定性：形成对角线的递归点占比 | **上升**（系统更规则） |
| **Lmax** | 最长对角线长度 | **延长**（周期结构） |
| **ENTR** | 对角线长度分布的香农熵 | **下降**（分布集中化） |
| **LAM** | 层流率：形成垂直线的递归点占比 | **上升**（陷阱态增多） |
| **TT** | 捕获时间：平均垂直线长度 | **上升**（更长时间处于同一状态） |

## 数学基础

| 理论 | 来源 | 作用 |
|------|------|------|
| Takens 嵌入定理 | Takens (1981) | 从标量时间序列重建相空间 |
| 递归图 RQA | Zbilut & Webber (1992) | 量化递归结构的 6 个指标 |
| 对角线/垂直线分析 | Marwan et al. (2007) | 区分周期、混沌、随机动力学 |

## 安装

```bash
pip install -r requirements.txt
# numpy, pandas, scipy
```

## 测试

```bash
cd tests
python test_rqa.py
# 11/11 tests passing
```

## 使用

```python
from rqa_factor import compute_rqa_factors

# 输入: T天 x N支股票的收益率矩阵
factors = compute_rqa_factors(returns, window=63, dim=3, tau=1)
# factors['det'], factors['rr'], factors['lmax'], factors['entr'], factors['lam'], factors['tt']

# 合成数据演示
cd notebooks
python demo_rqa.py
```

### 算法流程

1. **时滞嵌入** `phase_space_embedding(s, dim, tau)`：将每日收益率序列嵌入到 dim 维相空间
2. **递归矩阵** `recurrence_matrix(X, pct)`：计算相空间点之间的阈值化距离矩阵
3. **RQA 指标** `recurrence_quantification(R)`：从递归矩阵提取 6 个非线性动力学指标
4. **滚动计算** `compute_rqa_factors(returns, window)`：滑动窗口滚动计算

## 已知限制

- 多资产收益率被等权合并为一个市场序列，丢失了横截面信息
- RR 由固定百分位阈值控制（pct=10），因此方差几乎为零——这不是 bug，而是设计选择
- LAM 和 TT 对递归矩阵密度敏感，在小样本窗口（<50天）可能出现离群值

## 诚实声明

- 递归图 RQA 框架来自 Zbilut、Webber 和 Marwan 的经典非线性动力学文献，非原创
- 本仓库的核心价值在于：干净的 Python 工程实现 + 可复现的合成数据验证
- 如用此因子获利，请感谢非线性动力学社区——但更可能是亏损

## License

MIT
