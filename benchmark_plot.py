import csv
import matplotlib.pyplot as plt
import seaborn as sns
import sys

methods = []
times = []
speedups = []

try:
    with open('results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods.append(row['method'])
            times.append(float(row['median_time_seconds']))
            speedups.append(float(row['speedup_vs_naive']))
except FileNotFoundError:
    print("Error: results.csv not found. Please run the benchmark first.")
    sys.exit(1)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
bars = plt.bar(methods, times, color=colors[:len(methods)], edgecolor='black', alpha=0.9)

plt.title('High-Performance Matrix Multiplication Benchmark (1024x1024)', fontsize=14, fontweight='bold')
plt.ylabel('Median Execution Time (Seconds) - Lower is Better', fontsize=12)
plt.xlabel('Optimization Techniques', fontsize=12)
plt.ylim(0, max(times) * 1.15) 

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02 * max(times), f'{yval:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)

best_speedup = max(speedups)
best_method_idx = speedups.index(best_speedup)

plt.annotate(f'{best_speedup:.1f}x FASTER!', 
             xy=(best_method_idx, times[best_method_idx]), 
             xytext=(best_method_idx + 0.5, times[best_method_idx] + 0.2 * max(times)),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='green', fontweight='bold')

plt.savefig('benchmark_graph.png', dpi=300, bbox_inches='tight')
print("✅ Graph saved successfully as 'benchmark_graph.png'")