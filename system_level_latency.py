"""
Experiment 2: System-Level Availability
Attack Type: Resource Exhaustion (CPU Overload)
Mechanism: NMS (Non-Maximum Suppression) Worst-Case Execution
"""

import torch
import torchvision.ops as ops
import time
import matplotlib.pyplot as plt

def generate_fake_boxes(n):
    """Generates 'n' random bounding boxes and scores."""
    boxes = torch.rand(n, 4) * 1000
    # Ensure x2 > x1 and y2 > y1
    boxes[:, 2:] += boxes[:, :2]
    scores = torch.rand(n)
    return boxes, scores

def run_latency_test():
    # --- CONFIGURATION ---
    NUM_OBJECTS_CLEAN = 100    # Normal traffic
    NUM_OBJECTS_ATTACK = 50000 # Phantom objects attack

    print("⚠ Starting System Availability Stress Test...")

    # --- 1. NORMAL SCENARIO ---
    boxes_clean, scores_clean = generate_fake_boxes(NUM_OBJECTS_CLEAN)
    
    start_time = time.time()
    ops.nms(boxes_clean, scores_clean, iou_threshold=0.5)
    clean_latency_ms = (time.time() - start_time) * 1000
    print(f"🟢 Normal ({NUM_OBJECTS_CLEAN} boxes): {clean_latency_ms:.2f} ms")

    # --- 2. ATTACK SCENARIO ---
    boxes_attack, scores_attack = generate_fake_boxes(NUM_OBJECTS_ATTACK)
    
    start_time = time.time()
    ops.nms(boxes_attack, scores_attack, iou_threshold=0.5)
    attack_latency_ms = (time.time() - start_time) * 1000
    print(f"🔴 Attack ({NUM_OBJECTS_ATTACK} boxes): {attack_latency_ms:.2f} ms")

    # --- 3. IMPACT CALCULATION ---
    slowdown_factor = attack_latency_ms / clean_latency_ms
    print(f"\n🚨 SYSTEM IMPACT: {slowdown_factor:.1f}x latency increase")

    # --- 4. VISUALIZATION ---
    _plot_latency(clean_latency_ms, attack_latency_ms, slowdown_factor)

def _plot_latency(clean_ms, attack_ms, factor):
    labels = ['Normal', 'SlowTrack Attack']
    times = [clean_ms, attack_ms]
    colors = ['green', 'red']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, times, color=colors)

    plt.ylabel('Processing Time (ms)')
    plt.title(f'Availability Analysis: NMS Latency\n(Lag Increased by {factor:.0f}x)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, 
                 f"{yval:.1f} ms", ha='center', fontweight='bold')

    save_path = 'result_system_availability.png'
    plt.savefig(save_path)
    print(f"✅ Graph saved as '{save_path}'")
    # plt.show() # Uncomment if running in a notebook/GUI environment

if __name__ == "__main__":
    run_latency_test()
