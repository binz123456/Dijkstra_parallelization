import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

plt.figure(figsize=(6,4))
plt.bar(df['method'], df['ms'], color=['blue','green','orange'])
plt.ylabel('Time (ms)')
plt.title('Dijkstra Pathfinding Benchmark')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('speedup.png', dpi=200)
plt.show()
