"""
Global AI Jobs — Experience Group Comparison (Final Version)
模块：A 薪资分布 | B 统计量 | C 百分位 | D 回归趋势
      E job_role热力图 | F industry热力图
      I 风险对比 | J 技能&招聘 | K 雷达图 | L ANOVA检验
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────
DATA_PATH = "C:/Users/Lenovo/Desktop/6002/global_ai_jobs.csv"

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BINS   = [0, 2, 5, 10, float('inf')]
LABELS = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '11+ yrs']
COLORS = sns.color_palette("Blues_d", len(LABELS))
COLOR  = dict(zip(LABELS, COLORS))

# ─────────────────────────────────────────────────────
# 1. 加载 & 分箱
# ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df['exp_group'] = pd.cut(df['experience_years'],
                          bins=BINS, labels=LABELS,
                          right=True, include_lowest=True)

print(f"✓ 数据：{df.shape[0]} 行")
print("各组样本量：")
print(df['exp_group'].value_counts().sort_index().to_string())


# ═══════════════════════════════════════════════════════
# 模块 A：薪资分布对比（独立窗口 + 恢复彩色）
# ═══════════════════════════════════════════════════════
print("\n▶ 模块 A：薪资分布对比 - 彩色独立窗口")

# --- 窗口 1：KDE 密度分布图 (彩色) ---
plt.figure(figsize=(10, 6))
for grp in LABELS:
    data = df[df['exp_group'] == grp]['salary_usd'].dropna()
    # 关键点：color=COLOR[grp] 确保调用你定义的颜色字典
    sns.kdeplot(data, label=f'{grp} (n={len(data)})',
                color=COLOR[grp], linewidth=2.5, fill=True, alpha=0.15)

plt.title('Salary Density (KDE) by Experience Group', fontsize=13, fontweight='bold')
plt.xlabel('Salary (USD)')
plt.ylabel('Density')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show() 


# --- 窗口 2：水平箱线图 (彩色) ---
plt.figure(figsize=(10, 6))
# 准备数据列表
plot_data = [df[df['exp_group'] == g]['salary_usd'].dropna() for g in LABELS]

bp = plt.boxplot(
    plot_data,
    vert=False, 
    patch_artist=True, # 必须为 True 才能填充颜色
    medianprops=dict(color='black', linewidth=2.5),
    flierprops=dict(marker='o', markersize=3, alpha=0.4)
)

# 关键点：循环为每个 box 填充对应的颜色
for patch, grp in zip(bp['boxes'], LABELS):
    patch.set_facecolor(COLOR[grp]) # 绑定颜色字典
    patch.set_alpha(0.8)

plt.yticks(range(1, len(LABELS) + 1), LABELS, fontsize=11)
plt.title('A2 — Salary Boxplot by Experience Group', fontsize=13, fontweight='bold')
plt.xlabel('Salary (USD)')
plt.grid(True, alpha=0.3, linestyle='--', axis='x')
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 B：均值 / 中位数 / 标准差 条形对比
# ═══════════════════════════════════════════════════════
print("▶ 模块 B：薪资统计量对比")

stats_df = (df.groupby('exp_group', observed=True)['salary_usd']
            .agg(Mean='mean', Median='median', Std='std').loc[LABELS])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('B — Salary Statistics by Experience Group',
             fontsize=15, fontweight='bold')

bar_colors = [COLOR[g] for g in LABELS]
for ax, col, title in zip(axes,
                           ['Mean', 'Median', 'Std'],
                           ['Mean Salary', 'Median Salary', 'Std Dev']):
    bars = ax.bar(LABELS, stats_df[col], color=bar_colors,
                  edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, stats_df[col]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + stats_df[col].max() * 0.015,
                f'${val:,.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('USD')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('expB_salary_stats.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 C：薪资百分位对比（P25 / P50 / P75）
# ═══════════════════════════════════════════════════════
print("▶ 模块 C：薪资百分位对比")

pct_df = (df.groupby('exp_group', observed=True)['salary_usd']
          .quantile([0.25, 0.5, 0.75])
          .unstack()
          .loc[LABELS])
pct_df.columns = ['P25', 'P50 (Median)', 'P75']

x     = np.arange(len(LABELS))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, (col, alpha, hatch) in enumerate(
        zip(pct_df.columns, [0.5, 0.75, 1.0], ['/', '', '\\'])):
    ax.bar(x + i * width, pct_df[col], width,
           label=col, color=bar_colors,
           alpha=alpha, hatch=hatch,
           edgecolor='black', linewidth=0.8)

ax.set_xticks(x + width)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_title('C — Salary Percentiles (P25 / P50 / P75) by Experience Group',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Salary (USD)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('expC_salary_percentiles.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 D：各组内薪资增长趋势线（回归）
# ═══════════════════════════════════════════════════════
print("▶ 模块 D：各组内回归趋势线")

fig, axes = plt.subplots(1, len(LABELS), figsize=(20, 5), sharey=True)
fig.suptitle('D — Experience Years vs Salary Regression (per Group)',
             fontsize=14, fontweight='bold')

for ax, grp in zip(axes, LABELS):
    sub = df[df['exp_group'] == grp][['experience_years', 'salary_usd']].dropna()
    ax.scatter(sub['experience_years'], sub['salary_usd'],
               alpha=0.3, s=20, color=COLOR[grp])
    if len(sub) >= 10:
        s, intercept, r, p, _ = stats.linregress(
            sub['experience_years'], sub['salary_usd'])
        x_line = np.linspace(sub['experience_years'].min(),
                             sub['experience_years'].max(), 100)
        ax.plot(x_line, s * x_line + intercept,
                color='red', linewidth=2,
                label=f'R²={r**2:.2f}\n+${s:,.0f}/yr')
        ax.legend(fontsize=9)
    ax.set_title(grp, fontsize=12, fontweight='bold', color=COLOR[grp])
    ax.set_xlabel('Experience Years')
    ax.grid(True, alpha=0.3, linestyle='--')

axes[0].set_ylabel('Salary (USD)')
plt.tight_layout()
plt.savefig('expD_regression.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 E：经验组 × job_role 热力图
# ═══════════════════════════════════════════════════════
print("▶ 模块 E：经验组 × job_role 热力图")

top_roles  = df['job_role'].value_counts().head(8).index
pivot_role = df[df['job_role'].isin(top_roles)].pivot_table(
    values='salary_usd', index='exp_group',
    columns='job_role', aggfunc='mean', observed=True).loc[LABELS]

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot_role, annot=True, fmt=',.0f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white',
            annot_kws={'fontsize': 9}, ax=ax)
ax.set_title('E — Avg Salary Heatmap: Experience Group × Job Role',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Job Role')
ax.set_ylabel('Experience Group')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('expE_heatmap_role.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 F：经验组 × industry 热力图
# ═══════════════════════════════════════════════════════
print("▶ 模块 F：经验组 × industry 热力图")

top_ind   = df['industry'].value_counts().head(8).index
pivot_ind = df[df['industry'].isin(top_ind)].pivot_table(
    values='salary_usd', index='exp_group',
    columns='industry', aggfunc='mean', observed=True).loc[LABELS]

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot_ind, annot=True, fmt=',.0f', cmap='YlGnBu',
            linewidths=0.5, linecolor='white',
            annot_kws={'fontsize': 9}, ax=ax)
ax.set_title('F — Avg Salary Heatmap: Experience Group × Industry',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Industry')
ax.set_ylabel('Experience Group')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('expF_heatmap_industry.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 I：自动化风险 & 裁员风险对比
# ═══════════════════════════════════════════════════════
print("▶ 模块 I：自动化风险 & 裁员风险对比")

risk_df = (df.groupby('exp_group', observed=True)
           [['automation_risk', 'layoff_risk']].mean().loc[LABELS])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('I — Risk Metrics by Experience Group',
             fontsize=14, fontweight='bold')

for ax, col, title, cmap in zip(
        axes,
        ['automation_risk', 'layoff_risk'],
        ['Automation Risk', 'Layoff Risk'],
        ['Oranges', 'Reds']):
    risk_colors = sns.color_palette(cmap, len(LABELS))
    bars = ax.bar(LABELS, risk_df[col], color=risk_colors,
                  edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, risk_df[col]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + risk_df[col].max() * 0.015,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, risk_df[col].max() * 1.2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('expI_risk.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 J：技能需求 & 招聘难度对比
# ═══════════════════════════════════════════════════════
print("▶ 模块 J：技能需求 & 招聘难度对比")

skill_df = (df.groupby('exp_group', observed=True)
            [['skill_demand_score', 'hiring_difficulty_score']].mean().loc[LABELS])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('J — Skill Demand & Hiring Difficulty by Experience Group',
             fontsize=14, fontweight='bold')

for ax, col, title, color in zip(
        axes,
        ['skill_demand_score', 'hiring_difficulty_score'],
        ['Skill Demand Score', 'Hiring Difficulty Score'],
        ['steelblue', 'darkorange']):
    bars = ax.bar(LABELS, skill_df[col], color=color, alpha=0.8,
                  edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, skill_df[col]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + skill_df[col].max() * 0.015,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('expJ_skill_hiring.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 K：雷达图（综合指标）
# ═══════════════════════════════════════════════════════
print("▶ 模块 K：综合指标雷达图")

radar_cols   = ['skill_demand_score', 'career_growth_score',
                'job_security_score', 'work_life_balance_score',
                'company_rating', 'automation_risk']
radar_labels = ['Skill Demand', 'Career Growth', 'Job Security',
                'Work-Life Balance', 'Company Rating', 'Automation Risk']

radar_df   = df.groupby('exp_group', observed=True)[radar_cols].mean().loc[LABELS]
radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, len(LABELS), figsize=(20, 5),
                          subplot_kw=dict(polar=True))
fig.suptitle('K — Key Metrics Radar by Experience Group (Normalized)',
             fontsize=14, fontweight='bold')

for ax, grp in zip(axes, LABELS):
    vals = radar_norm.loc[grp].tolist() + [radar_norm.loc[grp].tolist()[0]]
    ax.plot(angles, vals, color=COLOR[grp], linewidth=2.5)
    ax.fill(angles, vals, color=COLOR[grp], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''])
    ax.set_ylim(0, 1)
    ax.set_title(grp, fontsize=12, fontweight='bold', pad=12, color=COLOR[grp])
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('expK_radar.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 L：ANOVA 显著性检验
# ═══════════════════════════════════════════════════════
print("\n▶ 模块 L：ANOVA 显著性检验")
print("=" * 60)

groups_data = [df[df['exp_group'] == g]['salary_usd'].dropna() for g in LABELS]
f_stat, p_val = stats.f_oneway(*groups_data)
print(f"  F-statistic = {f_stat:.2f}")
print(f"  p-value     = {p_val:.2e}")
print(f"  结论：{'各组薪资差异显著 ✓' if p_val < 0.05 else '差异不显著'}")

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    df_anova = df[['exp_group', 'salary_usd']].dropna()
    tukey = pairwise_tukeyhsd(df_anova['salary_usd'],
                               df_anova['exp_group'].astype(str))
    print("\n  Tukey HSD 两两比较：")
    print(tukey.summary())
except ImportError:
    print("\n  （安装 statsmodels 可获得 Tukey 两两比较：pip install statsmodels）")

print("=" * 60)


# ═══════════════════════════════════════════════════════
# 汇总表格
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("  FULL SUMMARY TABLE")
print("=" * 75)
summary = df.groupby('exp_group', observed=True)['salary_usd'].agg(
    Count='count',
    Mean='mean',
    Median='median',
    Std='std',
    P25=lambda x: x.quantile(0.25),
    P75=lambda x: x.quantile(0.75)
).loc[LABELS].round(0)
print(summary.to_string())
print("=" * 75)

print("\n✓ 已生成图表（保存在当前目录）：")
for ch, name in zip('ABCDEFIKJL',
                    ['薪资分布', '统计量', '百分位', '回归趋势',
                     'job_role热力图', 'industry热力图',
                     '风险对比', '技能&招聘', '雷达图', 'ANOVA']):
    print(f"   exp{ch}_*.png  ←  {name}")