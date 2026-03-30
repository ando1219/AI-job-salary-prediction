"""
Global AI Jobs — Region Group Comparison Analysis
自定义地区分组：欧美 / 亚洲 / USA / Singapore / Australia
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

# ── 核心：自定义地区分组 ──────────────────────────────
REGION_MAP = {
    'Brazil':      'Europe & Americas',
    'Canada':      'Europe & Americas',
    'France':      'Europe & Americas',
    'Germany':     'Europe & Americas',
    'Netherlands': 'Europe & Americas',
    'UK':          'Europe & Americas',
    'India':       'Asia',
    'Japan':       'Asia',
    'UAE':         'Asia',
    'USA':         'USA',
    'Singapore':   'Singapore',
    'Australia':   'Australia',
}

REGION_ORDER = ['USA', 'Singapore', 'Australia', 'Europe & Americas', 'Asia']
COLORS       = sns.color_palette("tab10", len(REGION_ORDER))
COLOR        = dict(zip(REGION_ORDER, COLORS))

# ─────────────────────────────────────────────────────
# 1. 加载数据 & 映射分组
# ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"✓ 数据：{df.shape[0]} 行 × {df.shape[1]} 列")

# 只保留 REGION_MAP 中的国家，映射成地区
df['region'] = df['country'].map(REGION_MAP)
df_f = df[df['region'].notna()].copy()

print(f"✓ 筛选后：{len(df_f)} 行")
print("\n各地区样本量：")
for r in REGION_ORDER:
    countries = [k for k, v in REGION_MAP.items() if v == r]
    n = (df_f['region'] == r).sum()
    print(f"  {r:<22} n={n:<6}  ({', '.join(countries)})")

N = len(REGION_ORDER)

# ═══════════════════════════════════════════════════════
# 模块 A：薪资分布对比（拆分为两个窗口）
# ═══════════════════════════════════════════════════════
print("\n▶ 模块 A：薪资分布对比 - 独立窗口展示")

# --- 窗口 1：KDE 密度图 ---
plt.figure(figsize=(10, 7)) # 创建第一个独立窗口
for r in REGION_ORDER:
    data = df_f[df_f['region'] == r]['salary_usd'].dropna()
    sns.kdeplot(data, label=f'{r} (n={len(data)})',
                color=COLOR[r], linewidth=2.5, fill=True, alpha=0.15)

plt.title('Salary Density (KDE) by Region', fontsize=14, fontweight='bold')
plt.xlabel('Salary (USD)')
plt.ylabel('Density')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.savefig('regA1_kde_distribution.png', dpi=150, bbox_inches='tight')
plt.show() # 显示第一个窗口

# --- 窗口 2：箱线图 ---
plt.figure(figsize=(10, 7)) # 创建第二个独立窗口
bp = plt.boxplot(
    [df_f[df_f['region'] == r]['salary_usd'].dropna() for r in REGION_ORDER],
    vert=False, patch_artist=True,
    medianprops=dict(color='black', linewidth=2.5),
    flierprops=dict(marker='o', markersize=3, alpha=0.4)
)

for patch, r in zip(bp['boxes'], REGION_ORDER):
    patch.set_facecolor(COLOR[r])
    patch.set_alpha(0.8)

plt.yticks(range(1, len(REGION_ORDER) + 1), REGION_ORDER, fontsize=11)
plt.title('A2 — Salary Boxplot by Region', fontsize=14, fontweight='bold')
plt.xlabel('Salary (USD)')
plt.grid(True, alpha=0.3, linestyle='--', axis='x')
plt.savefig('regA2_boxplot_distribution.png', dpi=150, bbox_inches='tight')
plt.show() # 显示第二个窗口


# ═══════════════════════════════════════════════════════
# 模块 B：均值 / 中位数 / 标准差 条形对比
# ═══════════════════════════════════════════════════════
print("▶ 模块 B：薪资统计量对比")

stats_df = (df_f.groupby('region')['salary_usd']
            .agg(Mean='mean', Median='median', Std='std')
            .loc[REGION_ORDER])

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('B — Salary Statistics by Region', fontsize=15, fontweight='bold')

bar_colors = [COLOR[r] for r in REGION_ORDER]
for ax, col, title in zip(axes,
                           ['Mean', 'Median', 'Std'],
                           ['Mean Salary', 'Median Salary', 'Std Dev']):
    bars = ax.barh(REGION_ORDER, stats_df[col],
                   color=bar_colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, stats_df[col]):
        ax.text(val + stats_df[col].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'${val:,.0f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('USD')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

plt.tight_layout()
plt.savefig('regB_salary_stats.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 C：薪资百分位对比（P25 / P50 / P75）
# ═══════════════════════════════════════════════════════
print("▶ 模块 C：薪资百分位对比")

pct_df = (df_f.groupby('region')['salary_usd']
          .quantile([0.25, 0.5, 0.75])
          .unstack()
          .loc[REGION_ORDER])
pct_df.columns = ['P25', 'P50', 'P75']

x     = np.arange(N)
width = 0.25
fig, ax = plt.subplots(figsize=(13, 6))
for i, (col, alpha, hatch) in enumerate(
        zip(['P25', 'P50', 'P75'], [0.5, 0.8, 1.0], ['/', '', '\\'])):
    ax.bar(x + i * width, pct_df[col], width,
           label=col, color=bar_colors, alpha=alpha,
           hatch=hatch, edgecolor='black', linewidth=0.7)
ax.set_xticks(x + width)
ax.set_xticklabels(REGION_ORDER, fontsize=11)
ax.set_title('C — Salary Percentiles (P25 / P50 / P75) by Region',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Salary (USD)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig('regC_salary_percentiles.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 D：各地区 experience_years vs salary 回归趋势
# ═══════════════════════════════════════════════════════
print("▶ 模块 D：各地区经验年限 vs 薪资回归趋势")

ncols = min(3, N)
nrows = (N + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(7 * ncols, 5 * nrows),
                          sharey=True)
axes = np.array(axes).flatten()
fig.suptitle('D — Experience Years vs Salary Regression by Region',
             fontsize=14, fontweight='bold')

for ax, r in zip(axes, REGION_ORDER):
    sub = df_f[df_f['region'] == r][['experience_years', 'salary_usd']].dropna()
    ax.scatter(sub['experience_years'], sub['salary_usd'],
               alpha=0.3, s=15, color=COLOR[r])
    if len(sub) >= 10:
        s, intercept, rv, p, _ = stats.linregress(
            sub['experience_years'], sub['salary_usd'])
        x_line = np.linspace(sub['experience_years'].min(),
                             sub['experience_years'].max(), 100)
        ax.plot(x_line, s * x_line + intercept,
                color='red', linewidth=2,
                label=f'R²={rv**2:.2f}\n+${s:,.0f}/yr')
        ax.legend(fontsize=9)
    ax.set_title(r, fontsize=12, fontweight='bold', color=COLOR[r])
    ax.set_xlabel('Experience Years')
    ax.grid(True, alpha=0.3, linestyle='--')

for ax in axes[N:]:
    ax.set_visible(False)
axes[0].set_ylabel('Salary (USD)')
plt.tight_layout()
plt.savefig('regD_regression.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 E：地区 × job_role 热力图
# ═══════════════════════════════════════════════════════
print("▶ 模块 E：地区 × job_role 热力图")

top_roles  = df['job_role'].value_counts().head(8).index
pivot_role = df_f[df_f['job_role'].isin(top_roles)].pivot_table(
    values='salary_usd', index='region',
    columns='job_role', aggfunc='mean').loc[REGION_ORDER]

fig_h = max(5, N * 0.7)
fig, ax = plt.subplots(figsize=(18, fig_h))
sns.heatmap(pivot_role, annot=True, fmt=',.0f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white',
            annot_kws={'fontsize': 9}, ax=ax)
ax.set_title('E — Avg Salary Heatmap: Region × Job Role',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Job Role')
ax.set_ylabel('Region')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('regE_heatmap_role.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 F：地区 × industry 热力图
# ═══════════════════════════════════════════════════════
print("▶ 模块 F：地区 × industry 热力图")

top_ind   = df['industry'].value_counts().head(8).index
pivot_ind = df_f[df_f['industry'].isin(top_ind)].pivot_table(
    values='salary_usd', index='region',
    columns='industry', aggfunc='mean').loc[REGION_ORDER]

fig, ax = plt.subplots(figsize=(18, fig_h))
sns.heatmap(pivot_ind, annot=True, fmt=',.0f', cmap='YlGnBu',
            linewidths=0.5, linecolor='white',
            annot_kws={'fontsize': 9}, ax=ax)
ax.set_title('F — Avg Salary Heatmap: Region × Industry',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Industry')
ax.set_ylabel('Region')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('regF_heatmap_industry.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 I：自动化风险 & 裁员风险对比
# ═══════════════════════════════════════════════════════
print("▶ 模块 I：自动化风险 & 裁员风险对比")

risk_df = (df_f.groupby('region')[['automation_risk', 'layoff_risk']]
           .mean().loc[REGION_ORDER])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('I — Risk Metrics by Region', fontsize=14, fontweight='bold')

for ax, col, title, cmap in zip(
        axes,
        ['automation_risk', 'layoff_risk'],
        ['Automation Risk', 'Layoff Risk'],
        ['Oranges', 'Reds']):
    risk_colors = sns.color_palette(cmap, N)
    bars = ax.barh(REGION_ORDER, risk_df[col],
                   color=risk_colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, risk_df[col]):
        ax.text(val + risk_df[col].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Score')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

plt.tight_layout()
plt.savefig('regI_risk.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 J：技能需求 & 招聘难度对比
# ═══════════════════════════════════════════════════════
print("▶ 模块 J：技能需求 & 招聘难度对比")

skill_df = (df_f.groupby('region')[['skill_demand_score', 'hiring_difficulty_score']]
            .mean().loc[REGION_ORDER])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('J — Skill Demand & Hiring Difficulty by Region',
             fontsize=14, fontweight='bold')

for ax, col, title, color in zip(
        axes,
        ['skill_demand_score', 'hiring_difficulty_score'],
        ['Skill Demand Score', 'Hiring Difficulty Score'],
        ['steelblue', 'darkorange']):
    bars = ax.barh(REGION_ORDER, skill_df[col],
                   color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, skill_df[col]):
        ax.text(val + skill_df[col].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Score')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

plt.tight_layout()
plt.savefig('regJ_skill_hiring.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 K：雷达图（各地区综合指标）
# ═══════════════════════════════════════════════════════
print("▶ 模块 K：综合指标雷达图")

radar_cols   = ['skill_demand_score', 'career_growth_score',
                'job_security_score', 'work_life_balance_score',
                'company_rating', 'automation_risk']
radar_labels = ['Skill Demand', 'Career Growth', 'Job Security',
                'Work-Life Balance', 'Company Rating', 'Automation Risk']

radar_df   = df_f.groupby('region')[radar_cols].mean().loc[REGION_ORDER]
radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
angles += angles[:1]

ncols = min(3, N)
nrows = (N + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(6 * ncols, 5 * nrows),
                          subplot_kw=dict(polar=True))
axes = np.array(axes).flatten()
fig.suptitle('K — Key Metrics Radar by Region (Normalized)',
             fontsize=14, fontweight='bold')

for ax, r in zip(axes, REGION_ORDER):
    vals = radar_norm.loc[r].tolist() + [radar_norm.loc[r].tolist()[0]]
    ax.plot(angles, vals, color=COLOR[r], linewidth=2.5)
    ax.fill(angles, vals, color=COLOR[r], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''])
    ax.set_ylim(0, 1)
    ax.set_title(r, fontsize=11, fontweight='bold', pad=12, color=COLOR[r])
    ax.grid(True, alpha=0.4)

for ax in axes[N:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('regK_radar.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════
# 模块 L：ANOVA 显著性检验
# ═══════════════════════════════════════════════════════
print("\n▶ 模块 L：ANOVA 显著性检验")
print("=" * 60)

groups_data = [df_f[df_f['region'] == r]['salary_usd'].dropna()
               for r in REGION_ORDER]
f_stat, p_val = stats.f_oneway(*groups_data)
print(f"  F-statistic = {f_stat:.2f}")
print(f"  p-value     = {p_val:.2e}")
print(f"  结论：{'各地区薪资差异显著 ✓' if p_val < 0.05 else '差异不显著'}")

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    df_anova = df_f[['region', 'salary_usd']].dropna()
    tukey = pairwise_tukeyhsd(df_anova['salary_usd'], df_anova['region'])
    print("\n  Tukey HSD 两两比较：")
    print(tukey.summary())
except ImportError:
    print("\n  （pip install statsmodels 可获得 Tukey 两两比较）")

print("=" * 60)


# ═══════════════════════════════════════════════════════
# 汇总表格
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  FULL SUMMARY TABLE — by Region")
print("=" * 80)
summary = df_f.groupby('region')['salary_usd'].agg(
    Count='count', Mean='mean', Median='median', Std='std',
    P25=lambda x: x.quantile(0.25),
    P75=lambda x: x.quantile(0.75)
).loc[REGION_ORDER].round(0)
print(summary.to_string())
print("=" * 80)

print("\n✓ 已生成图表：")
for ch, name in zip('ABCDEFIKJL',
                    ['薪资分布', '统计量', '百分位', '回归趋势',
                     'job_role热力图', 'industry热力图',
                     '风险对比', '技能&招聘', '雷达图', 'ANOVA']):
    print(f"   reg{ch}_*.png  ←  {name}")