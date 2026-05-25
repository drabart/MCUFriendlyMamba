#!/usr/bin/env python3
"""
Parse results from post_step directory and generate graphs for:
1. Total time taken for all models
2. Maximum memory used by the models (recorder data)
3. Time per operation for float full model
4. Time per operation for int8 full model
5. Memory per step in split model
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Results directory
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_file(filepath):
    """Parse a result file and extract time and memory data."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract total time
    time_match = re.search(r'Total time for all events: (\d+) us', content)
    total_time_us = int(time_match.group(1)) if time_match else None
    
    # Extract memory allocations (Arena allocation total)
    memory_matches = re.findall(r'\[RecordingMicroAllocator\] Arena allocation total (\d+) bytes', content)
    
    if memory_matches:
        # For split models: take max from 3 parts, for full models: take the only value
        max_memory = max(int(m) for m in memory_matches)
    else:
        max_memory = None
    
    return total_time_us, max_memory

def parse_operations(filepath):
    """Parse operation times from a full model file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the "Grouped Profiling Results" section for full models
    if 'Grouped Profiling Results' in content:
        section_start = content.find('Grouped Profiling Results')
        section_end = content.find('Stack High Water Mark', section_start)
        section = content[section_start:section_end]
        
        # Extract operation times
        operations = {}
        matches = re.findall(r'(\w+) took (\d+) us\.', section)
        for op_name, time in matches:
            operations[op_name] = int(time)
        
        return operations
    return {}

def parse_split_memory(filepath):
    """Parse memory per step from a split model file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    memory_steps = {}
    
    # Extract PreSSM memory
    pre_match = re.search(r'--- PreSSM Memory Allocation ---.*?\[RecordingMicroAllocator\] Arena allocation total (\d+) bytes', content, re.DOTALL)
    if pre_match:
        memory_steps['PreSSM'] = int(pre_match.group(1))
    
    # Extract StepSSM memory
    step_match = re.search(r'--- StepSSM Memory Allocation ---.*?\[RecordingMicroAllocator\] Arena allocation total (\d+) bytes', content, re.DOTALL)
    if step_match:
        memory_steps['StepSSM'] = int(step_match.group(1))
    
    # Extract PostSSM memory
    post_match = re.search(r'--- PostSSM Memory Allocation ---.*?\[RecordingMicroAllocator\] Arena allocation total (\d+) bytes', content, re.DOTALL)
    if post_match:
        memory_steps['PostSSM'] = int(post_match.group(1))
    
    return memory_steps

def main():
    """Main function to parse data and create graphs."""
    
    files = [
        'har_float_full.txt',
        'har_float_split.txt',
        'har_int8_full.txt',
        'har_int8_split.txt'
    ]
    
    data = {}
    for filename in files:
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            total_time, max_memory = parse_file(filepath)
            # Create readable label from filename
            label = filename.replace('_', ' ').replace('.txt', '').title()
            data[label] = {'time_us': total_time, 'memory_bytes': max_memory}
            print(f"{label}: Time={total_time} us, Memory={max_memory} bytes")
    
    # Extract data for plotting
    models = list(data.keys())
    times = [data[m]['time_us'] / 1000 for m in models]  # Convert to milliseconds
    memories = [data[m]['memory_bytes'] / 1024 for m in models]  # Convert to KB
    
    # Sort by time descending
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i], reverse=True)
    models_sorted = [models[i] for i in sorted_indices]
    times_sorted = [times[i] for i in sorted_indices]
    
    # Sort by memory descending
    memories_sorted = [memories[i] for i in sorted_indices]
    
    colors1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Graph 1: Total Time (separate figure)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(range(len(models_sorted)), times_sorted, color=colors1)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Inference Time by Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models_sorted)))
    ax1.set_xticklabels(models_sorted, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(times_sorted):
        ax1.text(i, v + max(times_sorted) * 0.02, f'{v:.1f} ms', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path1 = os.path.join(RESULTS_DIR, 'inference_time.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✓ Time graph saved to: {output_path1}")
    plt.close()
    
    # Graph 2: Maximum Memory (separate figure)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(range(len(models_sorted)), memories_sorted, color=colors1)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Memory (KB)', fontsize=12, fontweight='bold')
    ax2.set_title('Maximum Memory Used by Model', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models_sorted)))
    ax2.set_xticklabels(models_sorted, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(memories_sorted):
        ax2.text(i, v + max(memories_sorted) * 0.02, f'{v:.1f} KB', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path2 = os.path.join(RESULTS_DIR, 'memory_usage.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Memory graph saved to: {output_path2}")
    plt.close()
    
    # Graph 3: Time per operation for Float Full model
    float_full_path = os.path.join(RESULTS_DIR, 'har_float_full.txt')
    float_ops = parse_operations(float_full_path)
    if float_ops:
        # Sort by time descending
        sorted_ops = sorted(float_ops.items(), key=lambda x: x[1], reverse=True)
        ops = [op[0] for op in sorted_ops]
        times_ops = [op[1] for op in sorted_ops]
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        bars = ax3.bar(ops, [t/1000 for t in times_ops], color='#1f77b4')
        ax3.set_xlabel('Operation', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax3.set_title('Time per Operation - HAR Float Full Model', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f} ms', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path3 = os.path.join(RESULTS_DIR, 'ops_float_full.png')
        plt.savefig(output_path3, dpi=300, bbox_inches='tight')
        print(f"✓ Float Full operations graph saved to: {output_path3}")
        plt.close()
    
    # Graph 4: Time per operation for Int8 Full model
    int8_full_path = os.path.join(RESULTS_DIR, 'har_int8_full.txt')
    int8_ops = parse_operations(int8_full_path)
    if int8_ops:
        # Sort by time descending
        sorted_ops = sorted(int8_ops.items(), key=lambda x: x[1], reverse=True)
        ops = [op[0] for op in sorted_ops]
        times_ops = [op[1] for op in sorted_ops]
        
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        bars = ax4.bar(ops, [t/1000 for t in times_ops], color='#ff7f0e')
        ax4.set_xlabel('Operation', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax4.set_title('Time per Operation - HAR Int8 Full Model', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f} ms', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path4 = os.path.join(RESULTS_DIR, 'ops_int8_full.png')
        plt.savefig(output_path4, dpi=300, bbox_inches='tight')
        print(f"✓ Int8 Full operations graph saved to: {output_path4}")
        plt.close()
    
    # Graph 5: Memory per step in Split models (Float and Int8)
    float_split_path = os.path.join(RESULTS_DIR, 'har_float_split.txt')
    int8_split_path = os.path.join(RESULTS_DIR, 'har_int8_split.txt')
    
    float_split_memory = parse_split_memory(float_split_path)
    int8_split_memory = parse_split_memory(int8_split_path)
    
    if float_split_memory and int8_split_memory:
        # Keep order as PreSSM, StepSSM, PostSSM
        steps_order = ['PreSSM', 'StepSSM', 'PostSSM']
        float_mems = [float_split_memory.get(s, 0)/1024 for s in steps_order]  # Convert to KB
        int8_mems = [int8_split_memory.get(s, 0)/1024 for s in steps_order]  # Convert to KB
        
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        
        # Set up grouped bar positions
        x = np.arange(len(steps_order))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, float_mems, width, label='Float32', color='#1f77b4')
        bars2 = ax5.bar(x + width/2, int8_mems, width, label='Int8', color='#ff7f0e')
        
        ax5.set_xlabel('Model Step', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Memory (KB)', fontsize=12, fontweight='bold')
        ax5.set_title('Memory Usage per Step - Split Models (Float vs Int8)', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(steps_order)
        ax5.legend(fontsize=11)
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f} KB', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path5 = os.path.join(RESULTS_DIR, 'memory_split_steps.png')
        plt.savefig(output_path5, dpi=300, bbox_inches='tight')
        print(f"✓ Split model memory steps graph saved to: {output_path5}")
        plt.close()

if __name__ == '__main__':
    main()
