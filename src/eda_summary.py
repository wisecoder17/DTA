import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/amazon_processed.csv")

summary = []

for col in df.columns:
    col_type = df[col].dtype
    missing_pct = df[col].isna().mean() * 100
    
    if np.issubdtype(col_type, np.number):
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        min_ = df[col].min()
        max_ = df[col].max()
        notes = ""
    else:
        mean = median = std = min_ = max_ = np.nan
        levels = df[col].unique()
        notes = f"Levels: {levels[:10]}{'...' if len(levels) > 10 else ''}"

    summary.append({
        "Feature": col,
        "Type": "numeric" if np.issubdtype(col_type, np.number) else "categorical",
        "Missing (%)": round(missing_pct, 2),
        "Mean": round(mean, 2) if not pd.isna(mean) else "",
        "Median": round(median, 2) if not pd.isna(median) else "",
        "Std": round(std, 2) if not pd.isna(std) else "",
        "Min": round(min_, 2) if not pd.isna(min_) else "",
        "Max": round(max_, 2) if not pd.isna(max_) else "",
        "Notes": notes
    })

eda_summary = pd.DataFrame(summary)

# Save to CSV
eda_summary.to_csv("results/tables/eda_summary.csv", index=False)
print("EDA summary saved at results/tables/eda_summary.csv")
