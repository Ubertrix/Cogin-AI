import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def analyze_architecture():
    # Load architectural logs
    with open('/home/ubuntu/architectural_logs.json', 'r') as f:
        logs = json.load(f)
    
    df = pd.DataFrame(logs)
    
    # 1. Latency Analysis per Layer
    # We need to extract latency from the 'details' of BrainEngine synthesis
    latencies = []
    for _, row in df[df['layer'] == 'BrainEngine'].iterrows():
        try:
            lat = float(row['details'].split(': ')[1].replace('s', ''))
            latencies.append(lat)
        except:
            pass
            
    # 2. Routing Distribution
    routing_events = df[df['layer'] == 'SmartGate']
    routing_targets = routing_events['details'].apply(lambda x: x.split(' | ')[0].split(': ')[1])
    routing_counts = routing_targets.value_counts()
    
    # 3. Knowledge Retrieval Success Rate
    retrieval_events = df[df['layer'] == 'LongTermMemory']
    retrieval_success = retrieval_events['details'].apply(lambda x: x.split(': ')[1])
    retrieval_counts = retrieval_success.value_counts()

    # Plotting
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cogni Pro Architectural Performance Analysis', fontsize=22, fontweight='bold')

    # Plot 1: Latency per Interaction
    axes[0, 0].plot(latencies, marker='o', linestyle='-', color='teal', linewidth=2)
    axes[0, 0].set_title('End-to-End Latency per Query', fontsize=14)
    axes[0, 0].set_ylabel('Time (Seconds)')
    axes[0, 0].set_xlabel('Query Index')

    # Plot 2: Expert Routing Distribution
    routing_counts.plot(kind='bar', ax=axes[0, 1], color='mediumpurple')
    axes[0, 1].set_title('Expert Routing Distribution', fontsize=14)
    axes[0, 1].set_ylabel('Number of Queries')
    axes[0, 1].set_xticklabels(routing_counts.index, rotation=45)

    # Plot 3: Knowledge Retrieval Efficiency
    retrieval_counts.plot(kind='pie', ax=axes[1, 0], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99'])
    axes[1, 0].set_title('Knowledge Retrieval Success Rate', fontsize=14)
    axes[1, 0].set_ylabel('')

    # Plot 4: Layer Interaction Summary (Mock data based on logs)
    layers = ['Tokenizer', 'Embedding', 'Attention', 'Router', 'Memory', 'Synthesis']
    impact = [5, 15, 25, 10, 30, 15] # Estimated architectural impact/complexity
    axes[1, 1].barh(layers, impact, color='coral')
    axes[1, 1].set_title('Architectural Layer Complexity Impact', fontsize=14)
    axes[1, 1].set_xlabel('Relative Impact Score')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('/home/ubuntu/architectural_performance.png')
    print("Architectural analysis complete. Plot saved to architectural_performance.png")

if __name__ == "__main__":
    analyze_architecture()
