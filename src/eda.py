import pandas as pd
import numpy as np 
import seaborn as sns 
import preprocessing as prep
import matplotlib.pyplot as plt

df = prep.run_prep()
print("=== EDA STARTED ===")
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Set style
sns.set_style("whitegrid")

# 1. Categorical Data Analysis (FIXED - No warnings)
plt.figure(figsize=(15, 10))

# Plot 1: Countplot (already correct)
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='Type', hue='Machine failure', palette='Set2')
plt.title('Machine Failure Distribution by Product Type')

# Plot 2: Barplot - FIXED
plt.subplot(2, 2, 2)
failure_by_type = df.groupby('Type')['Machine failure'].mean().reset_index()
sns.barplot(data=failure_by_type, x='Type', y='Machine failure', 
            hue='Type', palette='viridis', legend=False)  # Added hue and legend=False
plt.title('Failure Rate by Product Type (%)')
plt.ylabel('Failure Rate')

# Plot 3: Top products - FIXED
plt.subplot(2, 2, 3)
top_products = df[df['Machine failure'] == 1]['Product ID'].value_counts().head(10)
top_products_df = pd.DataFrame({
    'Product': top_products.index,
    'Count': top_products.values
})
sns.barplot(data=top_products_df, x='Count', y='Product', 
            hue='Product', palette='Reds_r', legend=False)  # Added hue and legend=False
plt.title('Top 10 Products with Most Failures')
plt.xlabel('Failure Count')

# Plot 4: Total failures by type - FIXED
plt.subplot(2, 2, 4)
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
failure_by_type = df.groupby('Type')[failure_types].sum().sum(axis=1)
failure_by_type_df = failure_by_type.reset_index()
failure_by_type_df.columns = ['Type', 'Total_Failures']
sns.barplot(data=failure_by_type_df, x='Type', y='Total_Failures', 
            hue='Type', palette='coolwarm', legend=False)  # Added hue and legend=False
plt.title('Total Failures by Product Type')
plt.ylabel('Total Failure Count')

plt.tight_layout()
plt.show()

# 2. Continuous Variables Analysis
print("Creating continuous variable plots...")

# Scatter plots for continuous variables
continuous_pairs = [
    ('Air temperature [K]', 'Process temperature [K]'),
    ('Air temperature [K]', 'Torque [Nm]'),
    ('Process temperature [K]', 'Torque [Nm]'),
    ('Rotational speed [rpm]', 'Torque [Nm]')
]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (x_var, y_var) in enumerate(continuous_pairs):
    sns.scatterplot(data=df, x=x_var, y=y_var, hue='Machine failure',
                   palette={0: 'blue', 1: 'red'}, alpha=0.6, ax=axes[idx])
    axes[idx].set_title(f'{x_var} vs {y_var}')
    if idx > 0:  # Only show legend on first plot to avoid repetition
        axes[idx].get_legend().remove()

plt.tight_layout()
plt.show()

# 3. Distribution Analysis (FIXED - No warnings)
print("Creating distribution plots...")
plt.figure(figsize=(15, 10))

# Plot 1: Rotational speed distribution
plt.subplot(2, 3, 1)
sns.histplot(data=df, x='Rotational speed [rpm]', hue='Machine failure', 
             kde=True, alpha=0.6, palette={0: 'blue', 1: 'red'})
plt.title('Rotational Speed Distribution by Failure')

# Plot 2: Tool wear distribution
plt.subplot(2, 3, 2)
sns.histplot(data=df, x='Tool wear [min]', hue='Machine failure', 
             kde=True, alpha=0.6, palette={0: 'green', 1: 'orange'})
plt.title('Tool Wear Distribution by Failure')

# Plot 3: Tool wear boxplot
plt.subplot(2, 3, 3)
sns.boxplot(data=df, x='Machine failure', y='Tool wear [min]', palette='Set2')
plt.title('Tool Wear by Failure Status')

# Plot 4: Rotational speed violin plot
plt.subplot(2, 3, 4)
sns.violinplot(data=df, x='Machine failure', y='Rotational speed [rpm]', palette='pastel')
plt.title('Rotational Speed by Failure Status')

# Plot 5: Failure types - FIXED
plt.subplot(2, 3, 5)
failure_counts = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum().reset_index()
failure_counts.columns = ['Failure_Type', 'Count']
sns.barplot(data=failure_counts, x='Failure_Type', y='Count', 
            hue='Failure_Type', palette='Reds_r', legend=False)  # Added hue and legend=False
plt.title('Count of Different Failure Types')
plt.xticks(rotation=45)

# Plot 6: Scatter with size encoding
plt.subplot(2, 3, 6)
sns.scatterplot(data=df, x='Tool wear [min]', y='Rotational speed [rpm]', 
                hue='Machine failure', size='Machine failure',
                sizes={0: 20, 1: 60}, palette={0: 'gray', 1: 'red'}, alpha=0.6)
plt.title('Tool Wear vs Rotational Speed')

plt.tight_layout()
plt.show()

# 4. Correlation Heatmap
print("Creating correlation heatmap...")
numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
                'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            fmt='.2f',
            square=True,
            cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix of Sensor Metrics and Failures', fontsize=14, pad=20)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 5. Advanced Analysis - Pairplot
print("Creating pairplot...")
key_columns = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
               'Machine failure']

# Convert to categorical for better coloring
pairplot_df = df[key_columns].copy()
pairplot_df['Machine failure'] = pairplot_df['Machine failure'].astype('category')

sns.pairplot(pairplot_df, 
             hue='Machine failure',
             diag_kind='kde',
             palette={0: 'blue', 1: 'red'},
             plot_kws={'alpha': 0.6, 's': 30},
             corner=True)
plt.suptitle('Pairwise Relationships Colored by Machine Failure', y=1.02)
plt.show()
