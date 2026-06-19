from scipy.ndimage import rotate
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_accuracies(csv_paths, output_path=None, target_model=None):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
        
    dfs = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Error: File '{csv_path}' not found.")
            return
        try:
            df = pd.read_csv(csv_path, header=None)
            if len(df.columns) < 3:
                print(f"Error: The CSV file '{csv_path}' must contain at least three columns.")
                return
            df.rename(columns={0: 'model_name', 1: 'epoch', 2: 'accuracy'}, inplace=True)
            dfs.append(df)
        except Exception as e:
            print(f"An error occurred while processing the file '{csv_path}': {e}")
            return
            
    if not dfs:
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    if target_model:
        if isinstance(target_model, str):
            target_model = [target_model]
        df = df[df['model_name'].isin(target_model)]
        if df.empty:
            print(f"Error: No data found for models {target_model}.")
            return
        
        plt.figure(figsize=(10, 6))
        
        pivot_df = df.pivot_table(index='epoch', columns='model_name', values='accuracy', aggfunc='mean')
        
        ax = plt.gca()
        pivot_df.plot(kind='bar', ax=ax, width=0.8, alpha=0.9, edgecolor='black')
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}', 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='top', 
                            xytext=(0, -5), 
                            textcoords='offset points',
                            fontsize=9, color='white', fontweight='bold', rotation=90)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        if target_model:
            model_titles = ", ".join(target_model)
            plt.title(f'Evaluation Accuracy per Epoch')
        else:
            plt.title('Evaluation Accuracy per Epoch')
            
        plt.legend(title='Model Name')
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved successfully to '{output_path}'")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation accuracies from CSV files.")
    parser.add_argument("--csv-file", nargs='+', type=str, help="Path to the CSV file(s) containing accuracy data.", required=True)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-m", "--model-name", nargs='+', type=str, default=None, help="Name(s) of the model(s) to plot.")
    
    args = parser.parse_args()
    plot_accuracies(args.csv_file, args.output, args.model_name)

if __name__ == "__main__":
    main()
