# Training Plots

This directory stores visualizations generated during model training and evaluation.

## Generated Plots

The notebook automatically saves the following plots here:

- **Training/Validation Accuracy** - Accuracy curves over epochs
- **Training/Validation Loss** - Loss curves over epochs
- **Confusion Matrices** - BiLSTM and AraBERT confusion matrices
- **Model Comparison Charts** - Side-by-side accuracy and F1-score comparisons
- **Class Distribution** - Sentiment label distribution visualizations

## Usage

Plots are automatically saved when running the notebook cells. You can also manually save plots using:

```python
plt.savefig('results/training_plots/your_plot_name.png', dpi=300, bbox_inches='tight')
```
