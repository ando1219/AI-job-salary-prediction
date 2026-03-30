import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

# 环境配置
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
file_path = "C:/Users/Lenovo/Desktop/6002/global_ai_jobs.csv"
df = pd.read_csv(file_path)

# 定义原始特征类别
features_numeric     = ['experience_years', 'weekly_hours', 'skill_demand_score', 'ai_adoption_score']
features_categorical = ['job_role', 'ai_specialization', 'work_mode', 'industry', 'country']
target = 'salary_usd'

# 数据清洗
df = df[df[target] > 0]
for col in features_numeric:
    df[col] = df[col].fillna(df[col].median())
for col in features_categorical:
    df[col] = df[col].fillna(df[col].mode()[0])

# 执行独热编码
X_encoded = pd.get_dummies(df[features_categorical + features_numeric], columns=features_categorical)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# ==========================================
# 2. Hyperparameter Tuning (Gradient Boosting)
# ==========================================
print("🚀 Starting Gradient Boosting Grid Search...")

grid = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid={'learning_rate': [0.05, 0.1], 'n_estimators': [150, 200], 'max_depth': [4, 6]},
    cv=5, scoring='r2', n_jobs=-1
)
grid.fit(X_train, y_train)

model = grid.best_estimator_
print(f"✓ Best Params: {grid.best_params_}")

# 评估
y_pred = model.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
salary_mean = y_test.mean()
salary_median = y_test.median()
nrmse_mean = rmse / salary_mean
nrmse_median = rmse / salary_median

print("\n📌 RMSE Relative to Salary Level:")
print(f"Mean Salary   : ${salary_mean:,.2f}")
print(f"Median Salary : ${salary_median:,.2f}")
print(f"RMSE / Mean   : {nrmse_mean:.2%}")
print(f"RMSE / Median : {nrmse_median:.2%}")

print("\n🏆 Gradient Boosting Performance:")
print(f"   R² Score : {r2:.4f}")
print(f"   RMSE     : ${rmse:,.2f}")
print(f"   MAE     : ${mae:,.2f}")

# ==========================================
# 3. Aggregated Feature Importance
# ==========================================
def map_to_category(col_name):
    for original in features_categorical:
        if col_name.startswith(original + "_"):
            return original
    return col_name

imp_df = pd.DataFrame({
    'Feature':    X_encoded.columns,
    'Importance': model.feature_importances_
})
imp_df['Category'] = imp_df['Feature'].apply(map_to_category)

category_imp = (imp_df.groupby('Category')['Importance']
                .sum()
                .reset_index()
                .sort_values('Importance', ascending=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=category_imp, x='Importance', y='Category',
            palette='winter', hue='Category', legend=False)
plt.title('Aggregated Feature Importance by Category (Gradient Boosting)', fontsize=14)
plt.xlabel('Total Cumulative Importance')
plt.ylabel('Original Feature Category')
plt.tight_layout()
plt.show()

print("\n📊 Aggregated Category Importance Scores:")
print(category_imp.to_string(index=False))

# ==========================================
# 4. Learning Curve
# ==========================================
print("\nGenerating Learning Curve...")
train_sizes, train_scores, test_scores = learning_curve(
    model, X_encoded, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Score")
plt.plot(train_sizes, np.mean(test_scores,  axis=1), 'o-', color="g", label="Cross-validation Score")
plt.title('Learning Curve: Gradient Boosting', fontsize=14)
plt.xlabel('Training Size')
plt.ylabel('R² Score')
plt.legend(loc="best")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle='--')
plt.xlabel("Predicted Salary")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(y_test, bins=30, edgecolor='black', alpha=0.7)

plt.axvline(salary_mean, linestyle='--', linewidth=2, label=f"Mean = ${salary_mean:,.0f}")
plt.axvline(salary_median, linestyle='-.', linewidth=2, label=f"Median = ${salary_median:,.0f}")
plt.axvline(salary_mean + rmse, linestyle=':', linewidth=2, label=f"Mean + RMSE = ${salary_mean + rmse:,.0f}")
plt.axvline(salary_mean - rmse, linestyle=':', linewidth=2, label=f"Mean - RMSE = ${salary_mean - rmse:,.0f}")

plt.title("Salary Distribution with RMSE Reference", fontsize=14)
plt.xlabel("Salary (USD)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

# 理想预测线
plt.plot([min_val, max_val], [min_val, max_val], linewidth=2, label="Perfect Prediction")

# ±RMSE 误差带
plt.plot([min_val, max_val], [min_val + rmse, max_val + rmse], linestyle='--', linewidth=2, label="+RMSE")
plt.plot([min_val, max_val], [min_val - rmse, max_val - rmse], linestyle='--', linewidth=2, label="-RMSE")

plt.xlabel("Actual Salary (USD)")
plt.ylabel("Predicted Salary (USD)")
plt.title("Actual vs Predicted Salary with RMSE Band", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()


relative_error = np.abs(y_test - y_pred) / y_test * 100

plt.figure(figsize=(10, 5))
plt.scatter(y_test, relative_error, alpha=0.6)
plt.axhline(relative_error.mean(), linestyle='--', linewidth=2,
            label=f"Mean Relative Error = {relative_error.mean():.2f}%")

plt.xlabel("Actual Salary (USD)")
plt.ylabel("Absolute Percentage Error (%)")
plt.title("Relative Error vs Actual Salary", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

metrics = {
    'RMSE / Mean Salary': nrmse_mean * 100,
    'RMSE / Median Salary': nrmse_median * 100
}

plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values())
plt.ylabel("Percentage (%)")
plt.title("RMSE Relative to Salary Level", fontsize=14)

for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')

plt.tight_layout()
plt.show()