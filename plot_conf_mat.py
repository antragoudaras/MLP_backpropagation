import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def plot_confusion_matrix(confusion_matrix, method: str):
        conf_df = pd.DataFrame(confusion_matrix, index=[i for i in range(1, 11)], columns=[i for i in range(1, 11)])
        plt.figure(figsize=(10, 7))
        ax = sns.heatmap(conf_df, annot=True, fmt='g')
        ax.xaxis.tick_top()
        
        plt.yticks(rotation=0)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.title(f'Confusion Matrix of \n MLP with {method}')
        plt.tight_layout()
        
        folder_path = os.path.join(os.getcwd(), 'plots_assignment1')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(os.path.join(folder_path, f'confusion_matrix_{method}.png')):
            plt.savefig(os.path.join(folder_path, f'confusion_matrix_{method}.png'))
