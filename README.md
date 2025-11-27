# Comparative Analysis of 2D Adversarial Attacks in Autonomous Driving

This repository contains the experimental code for the research paper: **"Comparative Analysis of 2D Adversarial Attacks in Autonomous Driving: From Pixel-Level Integrity to System-Level Availability."**

## Overview

This research explores two distinct vectors of adversarial attacks on autonomous vehicle perception systems:

1.  **Pixel-Level Integrity (The "Lie"):** Using Projected Gradient Descent (PGD) to imperceptibly alter input images, causing misclassification in Deep Neural Networks (e.g., ResNet50).
2.  **System-Level Availability (The "Lag"):** Using the "SlowTrack" mechanism to flood the object detection pipeline (specifically Non-Maximum Suppression) with phantom bounding boxes, causing CPU spikes and dangerous latency.

## Repository Structure

- `src/pixel_level_pgd.py`: Script demonstrating PGD attacks on image classification.
- `src/system_level_latency.py`: Script demonstrating CPU resource exhaustion via NMS overload.

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
