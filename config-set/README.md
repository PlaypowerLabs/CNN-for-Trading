# CNN Training Configurations

This document contains all training configurations for the CNN trading models. Each configuration explores different hyperparameter combinations to optimize model performance.

## Configuration Overview

| Config | Description | Lookback Window | Batch Size | Epochs | Early Stop | Learning Rate | Weight Decay | Valid Ratio | Est. Time | File Link |
|--------|-------------|-----------------|------------|--------|-----------:|---------------|--------------|-------------|-----------|-----------|
| **Config 1** | Conservative Training | 60 | 32 | 40 | 8 | 0.0001 | 0.01 | 0.2 | 13.3h | [config-1.yml](./config-1.yml) |
| **Config 2** | Aggressive Learning | 60 | 64 | 45 | 12 | 0.0005 | 0.005 | 0.25 | 15.0h | [config-2.yml](./config-2.yml) |
| **Config 3** | High Regularization | 60 | 16 | 45 | 6 | 0.00001 | 0.05 | 0.35 | 15.0h | [config-3.yml](./config-3.yml) |
| **Config 4** | Fast Training | 60 | 128 | 30 | 8 | 0.001 | 0.008 | 0.15 | 10.0h | [config-4.yml](./config-4.yml) |
| **Config 5** | Balanced Medium | 60 | 64 | 40 | 10 | 0.0002 | 0.015 | 0.25 | 13.3h | [config-5.yml](./config-5.yml) |
| **Config 6** | Precision Focus | 60 | 256 | 25 | 6 | 0.002 | 0.003 | 0.1 | 8.3h | [config-6.yml](./config-6.yml) |
| **Config 7** | Overfitting Prevention | 60 | 8 | 50 | 15 | 0.000001 | 0.1 | 0.4 | 16.7h | [config-7.yml](./config-7.yml) |
| **Config 8** | Quick Convergence | 60 | 512 | 20 | 4 | 0.005 | 0.001 | 0.15 | 6.7h | [config-8.yml](./config-8.yml) |
| **Config 9** | Moderate Complexity | 60 | 32 | 45 | 12 | 0.0001 | 0.02 | 0.3 | 15.0h | [config-9.yml](./config-9.yml) |
| **Config 10** | Patient Training | 60 | 24 | 50 | 15 | 0.00005 | 0.03 | 0.28 | 16.7h | [config-10.yml](./config-10.yml) |

## Configuration Details

### Detailed Configuration Breakdown

| Config | Strategy | Key Characteristics | Use Case |
|--------|----------|-------------------|----------|
| **Config 1** | Conservative Training | Moderate LR (0.0001), balanced regularization | Stable baseline model |
| **Config 2** | Aggressive Learning | High LR (0.0005), low regularization | Fast experimentation |
| **Config 3** | High Regularization | Very low LR (0.00001), heavy weight decay (0.05) | Overfitting prevention |
| **Config 4** | Fast Training | High LR (0.001), large batch (128) | Quick prototyping |
| **Config 5** | Balanced Medium | Balanced parameters across dimensions | General-purpose training |
| **Config 6** | Precision Focus | Very high LR (0.002), huge batch (256) | Fast convergence testing |
| **Config 7** | Overfitting Prevention | Ultra-low LR (0.000001), max regularization | Robust model training |
| **Config 8** | Quick Convergence | Highest LR (0.005), massive batch (512) | Rapid proof-of-concept |
| **Config 9** | Moderate Complexity | Standard LR (0.0001), medium regularization | Production candidate |
| **Config 10** | Patient Training | Low LR (0.00005), high regularization | Careful feature learning |

### Strategy Groups

#### **🛡️ Conservative Approaches (Configs 1, 3, 7, 9, 10)**
- **Learning Rates**: 0.000001 - 0.0001 (careful updates)
- **Regularization**: Medium to high weight decay
- **Training Time**: 13-17 hours (patient approach)
- **Best For**: Production deployments, stable convergence

#### **⚡ Aggressive Approaches (Configs 2, 4, 6, 8)**
- **Learning Rates**: 0.0005 - 0.005 (rapid updates)
- **Regularization**: Low weight decay for maximum capacity
- **Training Time**: 7-15 hours (fast experimentation)
- **Best For**: Quick prototyping, hyperparameter exploration

#### **⚖️ Balanced Approach (Config 5)**
- **Learning Rate**: 0.0002 (moderate)
- **Regularization**: Medium weight decay (0.015)
- **Training Time**: 13.3 hours (balanced)
- **Best For**: Starting point for hyperparameter tuning

## Usage Instructions

To run training with any configuration:

```bash
python training.py --input-data-path /path/to/data --output-path /path/to/output --config-path /path/to/config-set
```

The training script will automatically:
1. Discover all `.yml` files in the config directory
2. Create separate output folders named after each config file
3. Train models with the specified hyperparameters
4. Save results and metadata for each configuration

## Model Architecture Mapping

All configurations use:
- **LOOKBACK_WIN: 60** → CNN60d model (4 conv blocks, 128×180 input)
- **LABEL: "RET5"** → 5-day return prediction target
- **Input channels: 1** (grayscale financial time series images)
- **Output classes: 2** (binary classification: up/down)

## Expected Training Times

| Config Type | Estimated Training Time* | GPU Memory Usage |
|-------------|-------------------------|------------------|
| Quick (4, 6, 8) | 6-17 hours | 2-4 GB |
| Moderate (1, 5, 9) | 13-15 hours | 1-3 GB |
| Patient (2, 3, 7, 10) | 15-17 hours | 0.5-2 GB |

*Times are based on 20 minutes per epoch. Actual times depend on dataset size and hardware specifications.

## Output Structure

Each configuration will generate:
```
output/
├── config-1/
│   ├── CNN60d_RET5.pth.tar          # Model checkpoint
│   ├── CNN60d_RET5.pth              # Clean weights for inference
│   ├── training_metrics.json        # Loss/accuracy history
│   └── execution_metadata.json      # Training metadata
├── config-2/
│   └── ...
└── ...
```

## Hyperparameter Rationale

- **Batch Size**: Ranges from 8-512 to explore memory vs. gradient quality trade-offs
- **Learning Rate**: Spans 4 orders of magnitude (0.000001 to 0.005) for comprehensive exploration  
- **Weight Decay**: Varies from light (0.001) to heavy (0.1) regularization
- **Validation Ratio**: 0.1-0.4 range to balance training data vs. evaluation reliability
- **Early Stopping**: 4-15 epochs patience to prevent overfitting while allowing convergence
- **Epochs**: Limited to 20-50 epochs (6.7-16.7 hours) for practical training times

## Training Time Summary

**Total estimated time for all 10 configurations: ~139 hours**

| Time Range | Configurations | Purpose |
|------------|----------------|---------|
| **6-10 hours** | Config 4, 6, 8 | Quick experimentation and prototyping |
| **13-15 hours** | Config 1, 2, 3, 5, 9 | Standard training runs |
| **16-17 hours** | Config 7, 10 | Patient training with heavy regularization |

## Best Practices

1. **Start with Config 5** (Balanced Medium) for initial experiments
2. **Use Config 8** (Quick Convergence) for rapid prototyping  
3. **Deploy Config 1 or 9** for production models (conservative approaches)
4. **Try Config 3 or 7** if experiencing overfitting issues
5. **Monitor early stopping** - training may complete before reaching max epochs

---

Generated on: November 10, 2025  
Model Architecture: CNN60d (4-layer ConvNet)  
Target: 5-day return prediction (RET5)