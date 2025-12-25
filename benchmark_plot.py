import matplotlib.pyplot as plt
import seaborn as sns


methods = ['Naive', 'Optimized', 'Tiled', 'AVX (Manual)', 'Parallel AVX']
times = [6.14, 0.55, 0.48, 0.79, 0.60] 

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))


colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
bars = plt.bar(methods, times, color=colors, edgecolor='black', alpha=0.9)

plt.title('High-Performance Matrix Multiplication Benchmark (1024x1024)', fontsize=14, fontweight='bold')
plt.ylabel('Execution Time (Seconds) - Lower is Better', fontsize=12)
plt.xlabel('Optimization Techniques', fontsize=12)
plt.ylim(0, 7) 


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval}s', ha='center', va='bottom', fontweight='bold', fontsize=10)


plt.annotate(f'12.8x FASTER!', 
             xy=(2, 0.48), xytext=(3, 3), # Arrow head at Tiled, Text away
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='green', fontweight='bold')

plt.savefig('benchmark_graph.png', dpi=300, bbox_inches='tight')
print("✅ Graph saved successfully as 'benchmark_graph.png'")