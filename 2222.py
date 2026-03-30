import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('global_ai_jobs.csv')

# 2. 特征工程
df['adjusted_salary'] = df['salary_usd'] / df['cost_of_living_index']
df['bonus_ratio'] = df['bonus_usd'] / df['salary_usd']
df['skill_value'] = df['skill_demand_score'] / (df['automation_risk'] + 1)

# ============================================================
# 全局配色（浅色背景，适合插入 PPT）
# ============================================================
BG_COLOR    = '#FFFFFF'   # 白色背景
PANEL_COLOR = '#F5F7FA'   # 浅灰面板
TEXT_COLOR  = '#1A1A2E'   # 深色文字
GRID_COLOR  = '#D8DEE9'   # 浅网格线

# 互补撞色保持不变
COLOR_A = '#F4A020'       # 暖橙
COLOR_B = '#0A90C4'       # 加深青蓝（浅背景下更清晰，避免过浅发虚）

plt.rcParams.update({
    'figure.facecolor':  BG_COLOR,
    'axes.facecolor':    PANEL_COLOR,
    'axes.edgecolor':    GRID_COLOR,
    'axes.labelcolor':   TEXT_COLOR,
    'xtick.color':       TEXT_COLOR,
    'ytick.color':       TEXT_COLOR,
    'text.color':        TEXT_COLOR,
    'grid.color':        GRID_COLOR,
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
})

# ---------------------------
# 图1：调整后的薪资
# ---------------------------
country_stats = df.groupby('country').agg({
    'adjusted_salary': 'mean',
    'bonus_ratio':     'mean',
    'skill_value':     'mean'
}).sort_values(by='adjusted_salary', ascending=False).head(15).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_COLOR)

bars = ax.barh(
    country_stats['country'],
    country_stats['adjusted_salary'],
    color=COLOR_A,
    edgecolor='none',
    height=0.65
)

# 反转 y 轴让最高的在最上面
ax.invert_yaxis()
ax.xaxis.grid(True)
ax.set_axisbelow(True)

ax.set_title(
    'Top 15 Countries: Adjusted Salary (Salary / Cost of Living Index)',
    fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=14
)
ax.set_xlabel('Purchasing Power Index', color=TEXT_COLOR)
ax.set_ylabel('Country', color=TEXT_COLOR)

# 数值标签
for bar in bars:
    w = bar.get_width()
    ax.text(
        w + ax.get_xlim()[1] * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.2f}",
        va='center', ha='left',
        fontsize=10, color=COLOR_A, fontweight='bold'
    )

plt.tight_layout()
plt.show()

# ---------------------------
# 图2：薪资结构（Base + Bonus）
# ---------------------------
country_stats2 = df.groupby('country').agg({
    'salary_usd': 'mean',
    'bonus_usd':  'mean'
}).reset_index()

country_stats2['bonus_ratio'] = country_stats2['bonus_usd'] / country_stats2['salary_usd']
country_stats2 = (country_stats2
                  .sort_values(by='salary_usd', ascending=False)
                  .head(15)
                  .reset_index(drop=True))   # ✅ 重置索引，避免标签错位

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG_COLOR)

x = range(len(country_stats2))

ax.bar(x, country_stats2['salary_usd'], color=COLOR_A, label='Base Salary', edgecolor='none')
ax.bar(x, country_stats2['bonus_usd'],  color=COLOR_B, label='Bonus',
       bottom=country_stats2['salary_usd'], edgecolor='none')

ax.set_xticks(x)
ax.set_xticklabels(country_stats2['country'], rotation=40, ha='right', fontsize=10)
ax.yaxis.grid(True)
ax.set_axisbelow(True)

ax.set_title('Salary Structure by Country (Base + Bonus)',
             fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=14)
ax.set_xlabel('Country', color=TEXT_COLOR)
ax.set_ylabel('Average Salary (USD)', color=TEXT_COLOR)

max_height = (country_stats2['salary_usd'] + country_stats2['bonus_usd']).max()
ax.set_ylim(0, max_height * 1.18)

# 数值标签（位置统一）
for pos, row in country_stats2.iterrows():
    total = row['salary_usd'] + row['bonus_usd']
    ax.text(
        pos,
        total + max_height * 0.02,
        f"{row['bonus_ratio']:.1%}",
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        color=COLOR_B           # 青蓝标签与 Bonus 色呼应
    )

# 图例
legend_handles = [
    mpatches.Patch(color=COLOR_A, label='Base Salary'),
    mpatches.Patch(color=COLOR_B, label='Bonus'),
]
ax.legend(handles=legend_handles, facecolor=PANEL_COLOR,
          edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=11)

plt.tight_layout()
plt.show()