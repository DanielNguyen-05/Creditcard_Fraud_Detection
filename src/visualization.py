import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Setting up a general style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
}

def plot_histogram(X, col_idx, bins=50, title=None, savepath=None):
    """
    Draw a histogram with gradient fill and modern styling.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    # Calculate histogram
    counts, edges, patches = ax.hist(X[:, col_idx], bins=bins, 
                                     edgecolor='white', linewidth=0.5,
                                     alpha=0.9)
    
    # Apply gradient color
    cm = plt.cm.viridis
    norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
    for count, patch in zip(counts, patches):
        patch.set_facecolor(cm(norm(count)))
    
    # Styling
    ax.set_title(title or f'Distribution of Feature {col_idx}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add statistics
    mean_val = np.mean(X[:, col_idx])
    median_val = np.median(X[:, col_idx])
    ax.axvline(mean_val, color=COLORS['danger'], linestyle='--', 
              linewidth=2, label=f'Mean: {mean_val:.2f}', alpha=0.8)
    ax.axvline(median_val, color=COLORS['success'], linestyle='--', 
              linewidth=2, label=f'Median: {median_val:.2f}', alpha=0.8)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

def plot_loss(loss_history, savepath=None):
    """
    Draw training loss with styling
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    
    epochs = range(1, len(loss_history) + 1)
    
    # Plot training loss with gradient color
    ax.plot(epochs, loss_history, linewidth=2.5, 
           color=COLORS['primary'], label='Training Loss',
           marker='o', markersize=4, markevery=max(1, len(epochs)//20))
    
    # Styling
    ax.set_title('Model Training Progress', fontsize=18, 
                fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
             shadow=True, fancybox=True)
    
    # Highlight the improvement area and best epoch.
    if len(loss_history) > 1:
        best_epoch = np.argmin(loss_history) + 1
        best_loss = np.min(loss_history)
        ax.axvline(best_epoch, color=COLORS['success'], 
                  linestyle=':', linewidth=2, alpha=0.5)
        ax.plot(best_epoch, best_loss, 'r*', markersize=15,
               label=f'Best: Epoch {best_epoch} (Loss={best_loss:.4f})')
        ax.fill_between(epochs, 0, loss_history, alpha=0.1, 
                        color=COLORS['primary'])
        ax.legend(loc='upper right', fontsize=11, frameon=True, 
                 shadow=True, fancybox=True)
    
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

def plot_roc(fprs, tprs, label=None, savepath=None):
    """
    Draw ROC curve with modern styling and gradient fill.
    """
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    
    # Calculate AUC
    auc_score = np.trapz(tprs, fprs)
    
    # Draw ROC curve
    ax.plot(fprs, tprs, color=COLORS['primary'], linewidth=3,
           label=label or f'ROC Curve (AUC = {auc_score:.3f})')
    
    # Fill area under curve with gradient
    ax.fill_between(fprs, 0, tprs, alpha=0.2, color=COLORS['primary'])
    
    # Diagonal
    ax.plot([0, 1], [0, 1], '--', lw=2, color='gray', 
           alpha=0.5, label='Random Classifier (AUC = 0.500)')
    
    # Styling
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curve Analysis', fontsize=18, 
                fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=12, frameon=True, 
             shadow=True, fancybox=True)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, labels=['0', '1'], normalize=False, savepath=None):
    """
    Draw a confusion matrix with modern styling and detailed annotations.
    """
    cm_original = cm.copy()
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Colormap
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    # Draw heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Proportion' if normalize else 'Count', 
                       rotation=-90, va="bottom", fontsize=12, fontweight='bold')
    
    # Set up ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm[i, j]
            if normalize:
                text = f'{value:.1%}\n({cm_original[i, j]})'
            else:
                text = f'{int(value)}'
            
            ax.text(j, i, text, ha="center", va="center",
                   fontsize=14, fontweight='bold',
                   color="white" if cm[i, j] > thresh else "black")
    
    # Add grid lines
    ax.set_xticks(np.arange(len(labels)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(labels)+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()