import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_losses(csv_path, output_path=None, target_model=None, window_size=20):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return

    try:
        df = pd.read_csv(csv_path, header=None)
        
        if len(df.columns) < 3:
            print("Error: The CSV file must contain at least three columns (Model name, Epoch, Loss value).")
            return
            
        df.rename(columns={0: 'model_name', 1: 'epoch', 2: 'loss'}, inplace=True)
        
        if target_model:
            df = df[df['model_name'] == target_model]
            if df.empty:
                print(f"Error: No data found for model '{target_model}'.")
                return
        
        model_col = 'model_name'
        epoch_col = 'epoch'
        loss_col = 'loss'
        
        plt.figure(figsize=(10, 6))

        for model_name, group in df.groupby(model_col):
            group = group.reset_index(drop=True)
            steps = group.index + 1
            
            plt.plot(steps, group[loss_col], linestyle='-', alpha=0.8, label=f"{model_name} (raw)")
            
            smoothed_loss = group[loss_col].rolling(window=window_size, min_periods=1).mean()
            plt.plot(steps, smoothed_loss, linestyle='-', label=f"{model_name} (smoothed)", linewidth=2)
            
            epoch_starts = group.drop_duplicates(subset=[epoch_col])
            
            for idx in epoch_starts.index:
                plt.axvline(x=idx + 1, color='gray', linestyle='--', alpha=0.5)
                
            plt.xticks(epoch_starts.index + 1, epoch_starts[epoch_col].astype(int))
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        if target_model:
            plt.title(f'Training Loss: {target_model}')
        else:
            plt.title('Training Loss')
            
        plt.legend(title='Model Name')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved successfully to '{output_path}'")
        else:
            plt.show()
            
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot training losses from a CSV file.")
    parser.add_argument("--csv-file", type=str, help="Path to the CSV file containing loss data.")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-m", "--model-name", type=str, default=None, help="Name of the model to plot.")
    parser.add_argument("-w", "--window", type=int, default=20, help="Window size for smoothing the loss curve.")
    
    args = parser.parse_args()
    plot_losses(args.csv_file, args.output, args.model_name, args.window)

if __name__ == "__main__":
    main()
