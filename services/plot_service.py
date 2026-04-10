import matplotlib.pyplot as plt
import os

def save_probability_plot(class_names, probabilities, save_path):
    plt.figure(figsize=(5, 4))
    plt.bar(class_names, probabilities)
    plt.title('Class Probabilities')
    plt.ylabel('Confidence')
    plt.ylim([0, 1])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()