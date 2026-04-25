import pandas as pd
import numpy as np
from scipy import stats
import sys

def analyze_results(filename):
    """Analyze results from a data file and print statistics."""
    df = pd.read_csv(filename)
    
    print(f"=== Analysis for {filename} ===")
    print(f"Average ctot: {df['ctot'].mean():.4f}")
    print()
    
    # Print average ctot depending on numObjects
    print("Average ctot by numObjects:")
    avg_by_objects = df.groupby('numObjects')['ctot'].mean()
    for num_objects, avg_ctot in avg_by_objects.items():
        print(f"  {num_objects} objects: {avg_ctot:.4f}")
    print()
    
    # Print distribution of numObjects
    print("Distribution of numObjects:")
    distribution = df['numObjects'].value_counts().sort_index()
    for num_objects, count in distribution.items():
        percentage = (count / len(df)) * 100
        print(f"  {num_objects} objects: {count} instances ({percentage:.1f}%)")
    print()
    
    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_results.py <file1> <file2>")
        print("Example: python analyze_results.py randomBlocks_ELS31.STOP.dat randomBlocks_GITTINS.STOP.dat")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]

# Analyze both files
df_els = analyze_results(file1)
df_gittins = analyze_results(file2)

# Perform t-test on ctot values
t_stat, p_value = stats.ttest_ind(df_els['ctot'], df_gittins['ctot'])

print("=== Statistical Comparison ===")
print(f"{file1} mean ctot: {df_els['ctot'].mean():.4f}")
print(f"{file2} mean ctot: {df_gittins['ctot'].mean():.4f}")
print(f"Difference: {df_els['ctot'].mean() - df_gittins['ctot'].mean():.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
if p_value < 0.05:
    print("Result: The difference is statistically significant (p < 0.05)")
else:
    print("Result: The difference is NOT statistically significant (p >= 0.05)")
print()

# Perform t-tests by numObjects
print("=== Statistical Comparison by numObjects ===")
all_num_objects = sorted(set(df_els['numObjects'].unique()) | set(df_gittins['numObjects'].unique()))
for num_objects in all_num_objects:
    els_data = df_els[df_els['numObjects'] == num_objects]['ctot']
    gittins_data = df_gittins[df_gittins['numObjects'] == num_objects]['ctot']
    
    if len(els_data) > 0 and len(gittins_data) > 0:
        t_stat_obj, p_value_obj = stats.ttest_ind(els_data, gittins_data)
        print(f"\n{num_objects} objects:")
        print(f"  {file1} mean: {els_data.mean():.4f} (n={len(els_data)})")
        print(f"  {file2} mean: {gittins_data.mean():.4f} (n={len(gittins_data)})")
        print(f"  Difference: {els_data.mean() - gittins_data.mean():.4f}")
        print(f"  T-statistic: {t_stat_obj:.4f}")
        print(f"  P-value: {p_value_obj:.6f}")
        if p_value_obj < 0.05:
            print(f"  Result: Significant (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
    else:
        print(f"\n{num_objects} objects: Insufficient data for comparison")
