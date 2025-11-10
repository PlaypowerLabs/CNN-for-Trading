# CNN Training Configurations

This document contains all training configurations for the CNN trading models. Each configuration explores different hyperparameter combinations to optimize model performance.

## Configuration Overview

| Config | Description | Lookback Window | Batch Size | Epochs | Learning Rate | Weight Decay | Valid Ratio | File Link |
|--------|-------------|-----------------|------------|--------|---------------|--------------|-------------|-----------|
| **Config 1** | Conservative Training | 60 | 32 | 100 | 0.0001 | 0.01 | 0.2 | [config-1.yml](./config-1.yml) |
| **Config 2** | Aggressive Learning | 60 | 64 | 300 | 0.0005 | 0.005 | 0.25 | [config-2.yml](./config-2.yml) |
| **Config 3** | High Regularization | 60 | 16 | 150 | 0.00001 | 0.05 | 0.35 | [config-3.yml](./config-3.yml) |
| **Config 4** | Fast Training | 60 | 128 | 80 | 0.001 | 0.008 | 0.15 | [config-4.yml](./config-4.yml) |
| **Config 5** | Balanced Medium | 60 | 64 | 120 | 0.0002 | 0.015 | 0.25 | [config-5.yml](./config-5.yml) |
| **Config 6** | Precision Focus | 60 | 256 | 60 | 0.002 | 0.003 | 0.1 | [config-6.yml](./config-6.yml) |
| **Config 7** | Overfitting Prevention | 60 | 8 | 400 | 0.000001 | 0.1 | 0.4 | [config-7.yml](./config-7.yml) |
| **Config 8** | Quick Convergence | 60 | 512 | 40 | 0.005 | 0.001 | 0.15 | [config-8.yml](./config-8.yml) |
| **Config 9** | Moderate Complexity | 60 | 32 | 180 | 0.0001 | 0.02 | 0.3 | [config-9.yml](./config-9.yml) |
| **Config 10** | Patient Training | 60 | 24 | 500 | 0.00005 | 0.03 | 0.28 | [config-10.yml](./config-10.yml) |

## Configuration Details

### Strategy Groups

#### **Conservative Approaches (Configs 1, 3, 7, 9, 10)**
- Lower learning rates for stable convergence
- Higher regularization to prevent overfitting
- Moderate to longer training periods
- Suitable for production deployments

#### **Aggressive Approaches (Configs 2, 4, 6, 8)**
- Higher learning rates for faster convergence
- Lower regularization for maximum learning capacity
- Optimized for quick experimentation cycles
- May require careful monitoring for overfitting

#### **Balanced Approaches (Config 5)**
- Moderate hyperparameters across all dimensions
- Good starting point for hyperparameter tuning
- Balanced trade-off between training speed and stability

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
| Quick (4, 6, 8) | 2-4 hours | 2-4 GB |
| Moderate (1, 5, 9) | 4-8 hours | 1-3 GB |
| Patient (2, 3, 7, 10) | 8-20 hours | 0.5-2 GB |

*Times are approximate and depend on dataset size and hardware specifications.

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
- **Early Stopping**: 5-30 epochs patience to prevent overfitting while allowing convergence

---

Generated on: November 10, 2025  
Model Architecture: CNN60d (4-layer ConvNet)  
Target: 5-day return prediction (RET5)