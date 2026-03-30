# 🤖 AI Salary Prediction & Market Analysis

> **dataset is visualized in https://ando1219.github.io/AI-job-salary-prediction/ai_salary_explorer.html**

A data-driven analysis of the **Global AI, Data Science & Tech Jobs Dataset (2020–2026)** to predict salaries and uncover strategic insights about the AI talent market.

---

# 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Exploration](#-dataset-exploration)
- [Feature Engineering](#-feature-engineering)
- [AI Model Design](#-ai-model-design)
- [Model Evaluation](#-model-evaluation)
- [Strategic Insights](#-strategic-insights)
- [Conclusions](#-conclusions)
- [Limitations](#-limitations)
- [Future Work](#-future-work)

---

# 🚀 Project Overview

Artificial Intelligence (AI) is rapidly transforming industries worldwide. As AI adoption grows, understanding **salary structures and talent distribution** becomes critical for:

- 👨‍💻 Job seekers planning career paths  
- 🏢 Companies designing competitive compensation  
- 📊 HR teams benchmarking AI salaries  

This project analyzes the **Global AI, Data Science & Tech Jobs Dataset (2020–2026)** and builds predictive models to uncover the key factors influencing AI salaries.

---

## 🎯 Objectives

The project aims to:

- Build a **machine learning model to predict AI salaries**
- Identify **key factors influencing compensation**
- Segment the **AI job market into meaningful clusters**
- Provide **actionable insights for talent strategy**

---

## 💼 Business Value

The analysis delivers practical benefits:

- ⚡ Faster job evaluation for AI professionals
- 📊 Salary benchmarking for HR teams
- 🧠 Data-driven hiring strategies for organizations

---

# 📊 Dataset Exploration

### Dataset Overview

| Attribute | Value |
|---|---|
| Dataset | Global AI, Data Science & Tech Jobs Dataset |
| Records | 90,000 |
| Features | 35 |
| Countries | 12 |
| Job Roles | 8 |
| Time Period | 2020–2026 |

The dataset contains salary data, skills, experience, and contextual job attributes across the global AI industry.

---

## 👍 Dataset Strengths

- **Large scale** dataset enabling reliable machine learning models  
- **Multi-dimensional features** including experience, company context, and market indicators  
- **Mixed data types** suitable for diverse modelling approaches  

---

## ⚠️ Dataset Limitations

- Limited **time-series depth** for trend analysis  
- Missing **equity / stock compensation data**  
- Limited **skill-level granularity**

---

# 🛠 Feature Engineering

To improve model performance and interpretability, several transformations were applied.

---

## Encoding Methods

| Feature Type | Examples | Encoding Method |
|---|---|---|
| Unordered Categorical | country, job_role | One-hot encoding |
| Ordered Categorical | education_level, company_size | Ordinal encoding |
| Numerical | salary, experience_years | Standardisation |

---

## Engineered Features

### Adjusted Salary

Adjusted Salary = Salary / Cost of Living Index


Allows fair cross-country salary comparison.

---

### Bonus Ratio
bonus_ratio = bonus_usd / salary_usd


Measures compensation structure efficiency.

---

### Skill Value
skill_value = skill_demand_score / (automation_risk + 1)


Identifies high-value talent with strong skills and lower automation risk.

---

# 🤖 AI Model Design

Two types of machine learning approaches were used:

1️⃣ **Regression Models** – Salary prediction  
2️⃣ **Clustering Models** – Job market segmentation  

---

## 1️⃣ Regression Models

### Model Development Pipeline

1. Data preprocessing and encoding  
2. Hyperparameter tuning via **Grid Search**  
3. Model evaluation using **RMSE and R²**  
4. Feature importance analysis  

---

### Model Comparison

| Model | RMSE | R² Score |
|---|---|---|
| Random Forest | 11,937 | 0.9258 |
| **Gradient Boosting (Best)** | **11,606** | **0.9299** |
| Baseline Model | 14,997 | 0.8829 |

The **Gradient Boosting model** achieved the best performance and was selected as the final predictive model.

---

## 2️⃣ K-Means Clustering

K-Means clustering was applied to segment the AI job market.

### Features Used

| Feature | Meaning |
|---|---|
| salary_usd | Economic return |
| experience_years | Professional experience |
| company_rating | Employer quality |
| career_growth_score | Career development potential |
| work_life_balance_score | Workload & lifestyle balance |

---

### Optimal Number of Clusters

The **Elbow Method** identified:
K = 4


---

### Cluster Profiles

| Cluster | Avg Salary | Experience | Market Share |
|---|---|---|---|
| Entry-Level Jobs | $74,967 | 3.97 years | 28.17% |
| Quality Employer Jobs | $76,602 | 4.33 years | 27.64% |
| Senior High-Pay Jobs | $150,584 | 14.48 years | 22.18% |
| Work-Life Balance Jobs | $76,398 | 4.31 years | 22.01% |

---

# 📈 Model Evaluation

## Gradient Boosting Performance

| Metric | Value |
|---|---|
| RMSE | 11,606 |
| R² Score | 0.9299 |

The model explains **~93% of salary variance**, indicating strong predictive power.

---

## Feature Importance

Top salary drivers identified:

| Factor | Importance |
|---|---|
| Experience Years | **57%** |
| Country | **35.3%** |
| Job Role | **7.6%** |

Other factors such as **company_rating** and **skill_demand_score** have limited predictive influence.

---

# 📖 Strategic Insights

The analysis produces three major insights about the AI job market.

---

## 1️⃣ Experience Matters Most (57%)

Experience is the strongest determinant of salary.

Salary increases significantly across experience groups:

- 0–2 years  
- 3–5 years  
- 6–10 years  
- 11+ years  

### Recommendation

Companies should implement **dual career tracks**:

- Management Track  
- Technical Expert Track (AI Architect, Chief Data Scientist)

---

## 2️⃣ Location is Strategy (35%)

Salary levels vary significantly across countries.

Highest-paying markets:

1. 🇺🇸 United States  
2. 🇸🇬 Singapore  
3. 🇦🇺 Australia  

Lower-cost talent markets:

- 🇮🇳 India  
- 🇧🇷 Brazil  
- 🇦🇪 UAE  

### Recommendation

Adopt **global hiring strategies**:

- High-end research → USA / Singapore  
- Cost-efficient scaling → India / Brazil

---

## 3️⃣ Skills Matter More Than Job Titles

Salary distributions across roles overlap heavily:

- Data Scientist  
- ML Engineer  
- NLP Engineer  

This indicates **skills outweigh titles**.

### Recommendation

HR strategies should prioritize:

- Skills-based hiring
- Flexible role definitions
- Cross-role talent pools

---

## 4️⃣ AI Talent Market Segments

The clustering analysis reveals four talent segments:

| Segment | Strategy |
|---|---|
| Senior High-Pay Jobs | Invest in retention |
| Entry-Level Jobs | Build future talent pipelines |
| Work-Life Balance Jobs | Differentiate employer brand |
| Quality Employer Jobs | Maintain stable workforce |

---

# 📌 Conclusions

Key takeaways from the project:

1️⃣ **Capability Over Title**  
Experience explains **57% of salary variation**.

2️⃣ **Location is Strategic**  
Geography accounts for **35.3% of salary differences**.

3️⃣ **Data-Driven HR**  
Analytics can significantly improve hiring and compensation strategies.

4️⃣ **Market Segmentation Matters**  
Different talent segments require tailored hiring strategies.

---

# ⚠️ Limitations

- Limited **time-series data**
- Missing **equity and stock compensation**
- Clustering based on **limited features**

---

# 🔮 Future Work

Potential improvements include:

- Integrating **longitudinal salary data**
- Modeling **total compensation packages**
- Exploring **deep learning models**
- Developing an **interactive salary benchmarking dashboard**

---

# 👨‍💻 Author

Bairu.
