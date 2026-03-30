"""
Global AI Jobs — Job Role Comparison Analysis
每个职位单独分析，汇总成并排对比图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────
DATA_PATH = "C:/Users/Lenovo/Desktop/6002/global_ai_jobs.csv"
MIN_SAMPLES = 30          # 样本量低于此值的职位跳过
TOP_N_ROLES  = 8          # 最多展示前 N 个职位（按样本量排序）

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 加载数据
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"✓ 数据加载完成：{df.shape[0]} 行 × {df.shape[1]} 列")

# 选取样本量 >= MIN_SAMPLES 的职位，取前 TOP_N_ROLES 个
role_counts = df['job_role'].value_counts()
valid_roles = role_counts[role_counts >= MIN_SAMPLES].head(TOP_N_ROLES).index.tolist()
print(f"✓ 有效职位（n≥{MIN_SAMPLES}）：{valid_roles}")

df_filtered = df[df['job_role'].isin(valid_roles)].copy()
n_roles = len(valid_roles)

# 颜色方案（每个职位一个颜色）
PALETTE = sns.color_palette("tab10", n_roles)
ROLE_COLOR = dict(zip(valid_roles, PALETTE))

# ═══════════════════════════════════════════════════════
# 图1：薪资分布对比（KDE + 箱线图）
# ═══════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════
# 薪资分布对比（拆分为两个独立窗口）
# ═══════════════════════════════════════════════════════

# --- 窗口 1：KDE 密度曲线 ---
plt.figure(figsize=(10, 7))
# 直接使用 plt 或获取当前轴
for role in valid_roles:
    data = df_filtered[df_filtered['job_role'] == role]['salary_usd'].dropna()
    sns.kdeplot(data, label=role, color=ROLE_COLOR[role], linewidth=2, fill=True, alpha=0.12)

plt.title('Salary Density (KDE) by Job Role', fontsize=14, fontweight='bold')
plt.xlabel('Salary (USD)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.legend(fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('fig1_kde_only.png', dpi=150, bbox_inches='tight')
plt.show()  # 弹出第一个窗口


# --- 窗口 2：水平箱线图 ---
plt.figure(figsize=(10, 7))

# 重新计算排序（确保逻辑独立）
role_order = (df_filtered.groupby('job_role')['salary_usd']
              .median().sort_values(ascending=True).index.tolist())

bp = plt.boxplot(
    [df_filtered[df_filtered['job_role'] == r]['salary_usd'].dropna() for r in role_order],
    vert=False, patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    flierprops=dict(marker='o', markersize=3, alpha=0.4)
)

for patch, role in zip(bp['boxes'], role_order):
    patch.set_facecolor(ROLE_COLOR[role])
    patch.set_alpha(0.75)

plt.yticks(range(1, len(role_order) + 1), role_order, fontsize=10)
plt.title('Salary Boxplot (sorted by median)', fontsize=14, fontweight='bold')
plt.xlabel('Salary (USD)', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--', axis='x')

plt.tight_layout()
plt.savefig('fig1_boxplot_only.png', dpi=150, bbox_inches='tight')
plt.show()  # 弹出第二个窗口
# ═══════════════════════════════════════════════════════
# 图2：薪资统计汇总对比（均值 / 中位数 / 标准差）
# ═══════════════════════════════════════════════════════
stats_df = (df_filtered.groupby('job_role')['salary_usd']
            .agg(Mean='mean', Median='median', Std='std', Count='count')
            .loc[valid_roles]
            .sort_values('Median', ascending=False))

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Salary Statistics by Job Role', fontsize=16, fontweight='bold')

colors = [ROLE_COLOR[r] for r in stats_df.index]

for ax, col, title in zip(axes2,
                           ['Mean', 'Median', 'Std'],
                           ['Mean Salary', 'Median Salary', 'Salary Std Dev']):
    bars = ax.barh(stats_df.index, stats_df[col], color=colors, edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, stats_df[col]):
        ax.text(val + stats_df[col].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'${val:,.0f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('USD', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('fig2_salary_stats.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════
# 图3：经验年限 vs 薪资（各职位回归线对比）
# ═══════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(14, 8))

for role in valid_roles:
    sub = df_filtered[df_filtered['job_role'] == role][['experience_years', 'salary_usd']].dropna()
    color = ROLE_COLOR[role]

    # 散点（透明度低，突出趋势线）
    ax3.scatter(sub['experience_years'], sub['salary_usd'],
                alpha=0.15, s=15, color=color)

    # 线性回归趋势线
    slope, intercept, r, p, _ = stats.linregress(sub['experience_years'], sub['salary_usd'])
    x_line = np.linspace(sub['experience_years'].min(), sub['experience_years'].max(), 100)
    ax3.plot(x_line, slope * x_line + intercept,
             color=color, linewidth=2.5,
             label=f'{role}  (R²={r**2:.2f}, +${slope:,.0f}/yr)')

ax3.set_title('Experience Years vs Salary — Trend Lines by Job Role',
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Experience Years', fontsize=12)
ax3.set_ylabel('Salary (USD)', fontsize=12)
ax3.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
ax3.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('fig3_experience_vs_salary.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════
# 图4：各职位 × 经验等级 平均薪资热力图
# ═══════════════════════════════════════════════════════
pivot = (df_filtered.pivot_table(
    values='salary_usd',
    index='job_role',
    columns='experience_level',
    aggfunc='mean'
).loc[valid_roles])

fig4, ax4 = plt.subplots(figsize=(14, 7))
sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white',
            annot_kws={'fontsize': 9},
            ax=ax4)
ax4.set_title('Average Salary Heatmap: Job Role × Experience Level',
              fontsize=14, fontweight='bold', pad=15)
ax4.set_xlabel('Experience Level', fontsize=11)
ax4.set_ylabel('Job Role', fontsize=11)
plt.tight_layout()
plt.savefig('fig4_heatmap_role_experience.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════
# 图5：各职位关键指标雷达图
# ═══════════════════════════════════════════════════════
radar_cols = ['skill_demand_score', 'career_growth_score',
              'job_security_score', 'work_life_balance_score',
              'company_rating', 'automation_risk']
radar_labels = ['Skill Demand', 'Career Growth',
                'Job Security', 'Work-Life Balance',
                'Company Rating', 'Automation Risk']

# 归一化到 0-1
radar_df = df_filtered.groupby('job_role')[radar_cols].mean().loc[valid_roles]
radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig5 = plt.figure(figsize=(18, 4 * ((n_roles + 3) // 4)))
fig5.suptitle('Job Role Radar: Key Metrics (Normalized)', fontsize=16, fontweight='bold')

for idx, role in enumerate(valid_roles):
    ax = fig5.add_subplot((n_roles + 3) // 4, 4, idx + 1, polar=True)
    values = radar_norm.loc[role].tolist() + [radar_norm.loc[role].tolist()[0]]
    color = ROLE_COLOR[role]

    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=7)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', ''], fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_title(role, fontsize=10, fontweight='bold', pad=10, color=color)
    ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('fig5_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════
# 文字汇总表格
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  JOB ROLE SALARY SUMMARY TABLE")
print("=" * 90)

summary = df_filtered.groupby('job_role')['salary_usd'].agg(
    Count='count',
    Mean='mean',
    Median='median',
    Std='std',
    Min='min',
    Max='max'
).loc[valid_roles].sort_values('Median', ascending=False).round(0)

print(summary.to_string())
print("=" * 90)

# 回归结果汇总
print("\n" + "=" * 90)
print("  EXPERIENCE → SALARY REGRESSION SUMMARY (per role)")
print("=" * 90)
print(f"{'Job Role':<30} {'Slope ($/yr)':>14} {'R²':>8} {'P-value':>12}")
print("-" * 90)
for role in valid_roles:
    sub = df_filtered[df_filtered['job_role'] == role][['experience_years', 'salary_usd']].dropna()
    s, i, r, p, _ = stats.linregress(sub['experience_years'], sub['salary_usd'])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"{role:<30} {s:>14,.0f} {r**2:>8.3f} {p:>12.2e}  {sig}")
print("=" * 90)

print("\n✓ 所有图表已保存至当前目录：")
print("   fig1_salary_distribution.png")
print("   fig2_salary_stats.png")
print("   fig3_experience_vs_salary.png")
print("   fig4_heatmap_role_experience.png")
print("   fig5_radar.png")