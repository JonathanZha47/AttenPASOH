import os
import numpy as np
import pandas as pd
import sys
import argparse
sys.path.append('/Users/jonathanzha/Desktop/AttenPASOH')
from utils.util import eval_metrix

# Function to calculate metrics
def calculate_metrics(true_label_path, pred_label_path):
    # Load true and predicted labels
    true_label = np.load(true_label_path)
    pred_label = np.load(pred_label_path)
    
    # Calculate metrics using the eval_matrix function
    mae, mape, mse, rmse, r2, l1, l2 = eval_metrix(true_label, pred_label)
    
    return mae, mape, mse, rmse, r2, l1, l2

# Function to run experiment analysis
def run_experiment_analysis(experiments_path, num_experiments):
    experiments = [f"Experiment{i+1}" for i in range(num_experiments)]
    
    # Set the precision to 6 decimal places
    pd.options.display.float_format = '{:.6f}'.format

    # Initialize lists to store metrics for each experiment
    mse_list, mae_list, mape_list, rmse_list, r2_list, l1_list, l2_list = [], [], [], [], [], [], []

    # Loop over each experiment
    for experiment in experiments:
        true_label_path = os.path.join(experiments_path, experiment, 'true_label.npy')
        pred_label_path = os.path.join(experiments_path, experiment, 'pred_label.npy')
        
        # Calculate metrics for the current experiment
        metrics = calculate_metrics(true_label_path, pred_label_path)
        
        mae, mape, mse, rmse, r2, l1, l2 = metrics
        mse_list.append(mse)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        r2_list.append(r2)
        l1_list.append(l1)
        l2_list.append(l2)

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'Experiment': experiments,
        'MSE': mse_list,
        'MAE': mae_list,
        'MAPE': mape_list,
        'RMSE': rmse_list,
        'R^2': r2_list,
        'L1 Error': l1_list,
        'L2 Error': l2_list
    })

    # Round all values to 6 decimal places
    metrics_df = metrics_df.round(6)

    # Calculate the mean values across all numeric columns (excluding 'Experiment')
    mean_values = metrics_df.drop(columns=['Experiment']).mean(axis=0)

    # Append the mean values to the DataFrame
    mean_values_df = pd.DataFrame(mean_values).transpose()
    mean_values_df['Experiment'] = 'Mean'
    metrics_df = pd.concat([metrics_df, mean_values_df], ignore_index=True)

    # Print the results
    print(metrics_df)

    # Save the results to a CSV file
    output_csv_path = os.path.join(experiments_path, 'metrics_summary.csv')
    metrics_df.to_csv(output_csv_path, index=False, float_format='%.6f')
    print(f"Metrics summary saved to {output_csv_path}")

# Main function to parse arguments
def main():
    parser = argparse.ArgumentParser(description="Experiment Analysis Script")
    
    parser.add_argument('--experiments_path', type=str, required=True, 
                        help="Path to the experiments directory")
    parser.add_argument('--num_experiments', type=int, default=10, 
                        help="Number of experiments to analyze (default: 10)")
    
    args = parser.parse_args()

    # Run the experiment analysis with the parsed arguments
    run_experiment_analysis(args.experiments_path, args.num_experiments)



if __name__ == "__main__":
    main()
