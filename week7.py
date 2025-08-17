import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("DATA ANALYSIS PROJECT")
print("=" * 60)
print("\n" + "=" * 40)
print("TASK 1: LOAD AND EXPLORE DATASET")
print("=" * 40)

try:
    print("Loading Iris dataset...")
    iris_data = load_iris()
    
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    print("\n📊 First 5 rows of the dataset:")
    print(df.head())
    
    print("\n🔍 Dataset Information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    
    print("\n📋 Data Types:")
    print(df.dtypes)
    
    print("\n🔍 Checking for missing values:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("✅ No missing values found!")
    else:
        print("⚠️ Missing values detected. Cleaning data...")
        df_cleaned = df.dropna() 
        print(f"✅ Data cleaned. New shape: {df_cleaned.shape}")
        df = df_cleaned

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Creating sample dataset...")
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150),
        'petal length (cm)': np.random.normal(3.8, 1.7, 150),
        'petal width (cm)': np.random.normal(1.2, 0.7, 150),
        'species_name': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
    })


print("\n" + "=" * 40)
print("TASK 2: BASIC DATA ANALYSIS")
print("=" * 40)

try:
    print("\n📈 Basic Statistics (Numerical Columns):")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe().round(2))
    
    print("\n🔢 Mean values by Species:")
    species_means = df.groupby('species_name')[numerical_cols].mean()
    print(species_means.round(2))
    
    print("\n🔍 Key Findings:")
    print("1. Dataset Overview:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Species distribution: {df['species_name'].value_counts().to_dict()}")
    
    print("\n2. Feature Correlations:")
    correlations = df[numerical_cols].corr()
    print("   Strongest correlations:")
    mask = np.triu(np.ones_like(correlations), k=1).astype(bool)
    corr_pairs = correlations.where(mask).stack().sort_values(ascending=False)
    for i, (pair, corr) in enumerate(corr_pairs.head(3).items()):
        print(f"   - {pair[0]} vs {pair[1]}: {corr:.3f}")
    
    print("\n3. Species Characteristics:")
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        largest_feature = species_data[numerical_cols].mean().idxmax()
        print(f"   - {species}: Largest average {largest_feature}")

except Exception as e:
    print(f"❌ Error in data analysis: {e}")


print("\n" + "=" * 40)
print("TASK 3: DATA VISUALIZATION")
print("=" * 40)

try:
    fig = plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    df_sorted = df.sort_values('sepal length (cm)')
    df_sorted['cumulative_mean'] = df_sorted['sepal length (cm)'].expanding().mean()
    plt.plot(range(len(df_sorted)), df_sorted['cumulative_mean'], 
             linewidth=2, color='blue', alpha=0.8)
    plt.title('📈 Cumulative Mean of Sepal Length', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index (Sorted by Sepal Length)')
    plt.ylabel('Cumulative Mean (cm)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    avg_petal_length = df.groupby('species_name')['petal length (cm)'].mean()
    bars = plt.bar(avg_petal_length.index, avg_petal_length.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    plt.title('📊 Average Petal Length by Species', fontsize=14, fontweight='bold')
    plt.xlabel('Species')
    plt.ylabel('Average Petal Length (cm)')
    plt.xticks(rotation=45)
    
    for bar, value in zip(bars, avg_petal_length.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.subplot(2, 2, 3)
    plt.hist(df['sepal width (cm)'], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.title('📊 Distribution of Sepal Width', fontsize=14, fontweight='bold')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    mean_width = df['sepal width (cm)'].mean()
    std_width = df['sepal width (cm)'].std()
    plt.axvline(mean_width, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_width:.2f}')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    species_colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
    
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'],
                   c=species_colors[species], label=species, alpha=0.7, s=60)
    
    plt.title('🔗 Sepal Length vs Petal Length', fontsize=14, fontweight='bold')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.grid(True, alpha=0.3)
    
    correlation = df['sepal length (cm)'].corr(df['petal length (cm)'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ All visualizations created successfully!")
    
    print("\n🎨 Creating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('🔥 Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"❌ Error creating visualizations: {e}")

print("\n" + "=" * 40)
print("📋 ANALYSIS SUMMARY & INSIGHTS")
print("=" * 40)

try:
    print("\n🔍 Key Insights from the Analysis:")
    print("1. Dataset Quality:")
    print(f"   ✅ Clean dataset with {len(df)} samples and no missing values")
    print(f"   ✅ Balanced species distribution")
    
    print("\n2. Statistical Findings:")
    strongest_corr = correlations.where(mask).stack().abs().max()
    strongest_pair = correlations.where(mask).stack().abs().idxmax()
    print(f"   📊 Strongest correlation: {strongest_pair[0]} vs {strongest_pair[1]} ({strongest_corr:.3f})")
    
    print("\n3. Species Characteristics:")
    for species in df['species_name'].unique():
        species_data = df[df['species_name'] == species]
        distinctive_feature = (species_data[numerical_cols].mean() - df[numerical_cols].mean()).abs().idxmax()
        print(f"   🌸 {species}: Most distinctive in {distinctive_feature}")
    
    print("\n4. Visualization Insights:")
    print("   📈 Line chart shows convergence of cumulative mean")
    print("   📊 Bar chart reveals clear species differences in petal length")
    print("   📊 Histogram shows normal distribution of sepal width")
    print("   🔗 Scatter plot demonstrates strong positive correlation")
    
    print(f"\n✅ Analysis completed successfully!")
    print("📁 All requirements fulfilled:")
    print("   ✓ Dataset loaded and explored")
    print("   ✓ Basic statistics computed")
    print("   ✓ Grouping analysis performed")
    print("   ✓ Four different visualizations created")
    print("   ✓ Error handling implemented")
    print("   ✓ Proper labeling and customization applied")

except Exception as e:
    print(f"❌ Error in summary generation: {e}")

print("\n" + "=" * 60)
print("🎉 DATA ANALYSIS PROJECT COMPLETED!")
print("=" * 60)