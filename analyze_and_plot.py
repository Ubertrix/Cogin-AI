import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_and_plot():
    # Load results
    with open('/home/ubuntu/benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # 1. Average Confidence per Category
    avg_confidence = df.groupby('expected_category')['confidence'].mean()
    
    # 2. Average Latency per Category
    avg_latency = df.groupby('expected_category')['latency'].mean()
    
    # 3. Word Count per Category
    avg_words = df.groupby('expected_category')['word_count'].mean()

    # Plotting
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cogni Pro Performance Analysis', fontsize=20)

    # Plot 1: Confidence Level
    avg_confidence.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Average Confidence Level')
    axes[0, 0].set_ylabel('Confidence (0-1)')
    axes[0, 0].set_ylim(0, 1.1)

    # Plot 2: Latency
    avg_latency.plot(kind='bar', ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Average Latency (Seconds)')
    axes[0, 1].set_ylabel('Time (s)')

    # Plot 3: Word Count (Response Depth)
    avg_words.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Average Response Length (Words)')
    axes[1, 0].set_ylabel('Word Count')

    # Plot 4: Overall Distribution (Pie Chart)
    category_counts = df['expected_category'].value_counts()
    category_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', startangle=140, colors=['gold', 'orchid', 'cyan', 'tomato'])
    axes[1, 1].set_title('Test Case Distribution')
    axes[1, 1].set_ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('/home/ubuntu/performance_analysis.png')
    print("Analysis complete. Plot saved to performance_analysis.png")

    # Summary Statistics
    summary = {
        "overall_avg_confidence": df['confidence'].mean(),
        "overall_avg_latency": df['latency'].mean(),
        "total_test_cases": len(df),
        "category_performance": avg_confidence.to_dict()
    }
    
    with open('/home/ubuntu/performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    analyze_and_plot()
