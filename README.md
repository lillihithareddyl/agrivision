---
title: AgriVision AI
emoji: 🌿
colorFrom: green
colorTo: lime
sdk: docker
app_port: 8501
---

# AgriVision AI

A research prototype for leaf disease classification and fruit quality
classification using MobileNetV2.

## Models

- Leaf: 38 PlantVillage classes.
- Fruit: Good, Bad, and Mixed FruitNet quality labels.

## Limitations

Reported metrics are dataset-specific. They do not guarantee performance
on independent real-world images. The fruit model classifies quality labels
and does not localize defects. Predictions should not replace expert
agricultural assessment.
