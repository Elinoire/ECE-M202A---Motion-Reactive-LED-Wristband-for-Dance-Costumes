---
layout: default
title: Motion-Reactive LED Wristband for Dance Costumes
---

# ECE-M202A---Motion-Reactive-LED-Wristband-for-Dance-Costumes

# Motion-Reactive LED Wristband

An Arduino Nano 33 BLE Sense Rev2 worn on the wrist streams IMU data, and a classifier that distinguishes different dance moves, and lights an APA102 LED strip in a different color for each chosen movement.

**Status:** data collection and offline classification work (99.1% accuracy on unseen sessions), and on-board classification with LED output work.

## Final Scope
1. LED strip turns on
2. Collection of training data for specific dance moves (vs. general directions)
3. Classifier correctly identifies movement corresponding to IMU data

## Minimum Deliverable
A system that can be used during dance and other art performances, that synchronizes the colors of a LED light strip to specific movements.

## Evaluation Plan
1. Baseline: Simple threshold/rule-based classifier on IMU values
2. Primary metric: Classification accuracy on labeled trials per class, plus LED response latency (ms).
3. Stress condition: Faster or noisier movement.
4. Missing: Labeled motion data for specific dance moves (need to collect ourselves)

## Remaining Risks
1. Design is not wearable
2. High Latency

## Response to Feedback
**1. When doing continuous dance moves, how is the model classification accuracy, especially considering the 0.56s delay?**

We are working to improve our latency and have decided to work on specific dance moves for lighting changes, so the changes will not happen for every step in the dance.

**2. Consider implementing an overlapping sliding window with a 5-to-10 sample stride and check latency versus accuracy:**

We already use  a sliding window with a 10 sample stride, but this could be further reduced to a 5 window stride. For the question of latency we will try training on transitions, lower delay to confirm, use smaller windows, etc. We are actively working on tuning our accuracy vs. latency. One change that has already been made is to preemptively drop the use of Bluetooth for the project as the data transmission for IMU data was 0.5s, and we expect similar timings for other data transmission.


### Link to Updated Slides
https://docs.google.com/presentation/d/1O7Qg9AKKq4t3O7V1BodWmiS7-Hkh3Znb/edit?slide=id.p1#slide=id.p1

