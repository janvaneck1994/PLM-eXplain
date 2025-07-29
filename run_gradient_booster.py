import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# Constant: number of crafted/known features
CRAFTED_FEATURES = 34

task_labels = {
    0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K',
    9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T',
    17: 'V', 18: 'W', 19: 'Y',
    20: 'SS3_H', 21: 'SS3_E', 22: 'SS3_C',
    23: 'SS8_H', 24: 'SS8_E', 25: 'SS8_G', 26: 'SS8_I', 27: 'SS8_B',
    28: 'SS8_T', 29: 'SS8_S', 30: 'SS8_-',
    31: 'ASA',
    32: 'GRAVY',
    33: 'AROM'
}

# Store metrics for final summary
results_data = []

def create_plots(embedding_path, sequence_path, output_path, plot_title, adapted, custom_gradient):
    # --- Remap embedding names in title ---
    plot_title = (
        plot_title
        .replace("Original Embeddings", "ESM2 Embedding")
        .replace("Adapted Embeddings",  "Partitioned Embedding")
        .replace("Baseline Embeddings", "Informed Embedding")
    )

    # Determine embedding label for metrics summary
    if "ESM2 Embedding" in plot_title:
        embedding_label = "ESM2"
    elif "Partitioned Embedding" in plot_title:
        embedding_label = "Partitioned"
    else:
        embedding_label = "Informed"

    # Extract task name from the title
    task_name = plot_title.split(" Prediction")[0]

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set larger global font sizes for all plots
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })

    # Load embeddings and labels
    train_embeddings = pd.read_csv(f"downstream_task_data/new_embedding/train_{embedding_path}_embeddings.csv").to_numpy()
    # train_embeddings[:, 26] = 0
    test_embeddings = pd.read_csv(f"downstream_task_data/new_embedding/test_{embedding_path}_embeddings.csv").to_numpy()
    # test_embeddings[:, 26] = 0
    train_df = pd.read_csv(f"downstream_task_data/train{sequence_path}.csv")
    test_df = pd.read_csv(f"downstream_task_data/test{sequence_path}.csv")
    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()
    
    # Scale the data
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Train Gradient Boosting Classifier
    print(f"Training Gradient Boosting Classifier for {plot_title}...")
    gb = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(train_embeddings, train_labels_encoded)

    # Evaluate the model
    print(f"Evaluating Gradient Boosting Classifier for {plot_title}...")
    test_preds = gb.predict(test_embeddings)
    accuracy = accuracy_score(test_labels_encoded, test_preds)
    precision = precision_score(test_labels_encoded, test_preds)
    recall = recall_score(test_labels_encoded, test_preds)
    f1 = f1_score(test_labels_encoded, test_preds)
    test_probs = gb.predict_proba(test_embeddings)[:, 1]
    auroc = roc_auc_score(test_labels_encoded, test_probs)
    
    # Add results to our collection for summary table
    results_data.append({
        'Task': task_name,
        'Embedding Type': embedding_label,
        'ROC AUC': auroc,
        'Accuracy': accuracy,
        'F1 Score': f1
    })

    # Save scores to file
    scores_file = os.path.join(output_path, "scores.txt")
    with open(scores_file, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"AUROC: {auroc:.4f}\n")

    # SHAP Analysis
    print(f"Calculating SHAP values for {plot_title}...")
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(test_embeddings)
    
    # Handle binary vs. regression shap output
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals_to_plot = shap_values[1]
    else:
        shap_vals_to_plot = shap_values

    # Create feature names
    if adapted:
        feature_names = [task_labels[i] if i in task_labels else f"feature_{i}" \
                         for i in range(test_embeddings.shape[1])]
    else:
        feature_names = [f"feature_{i}" for i in range(test_embeddings.shape[1])]

    # SHAP summary plot
    plt.figure(figsize=(6, 16))
    shap.summary_plot(shap_vals_to_plot, features=test_embeddings, feature_names=feature_names,
                      max_display=10, show=False)
    # plt.title(plot_title)
    plot_path = os.path.join(output_path, "plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Calculate mean absolute SHAP values
    mean_shap_values = np.abs(shap_vals_to_plot).mean(axis=0)

    # DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Index': range(test_embeddings.shape[1]),
        'Feature': feature_names,
        'Importance': mean_shap_values
    }).sort_values(by="Importance", ascending=False)

    top_features = importance_df.head(10)
    
    # Bar colors for adapted vs. other
    if adapted:
        bar_colors = ["#6C8EBF" if idx < CRAFTED_FEATURES else "#F2918C" for idx in top_features["Index"]]
    else:
        bar_colors = ["gray"] * len(top_features)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["Feature"], top_features["Importance"], color=bar_colors)
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title(f"SHAP Feature Importance\n({plot_title})")
    ax.invert_yaxis()

    plt.tight_layout()
    feature_importance_path = os.path.join(output_path, "feature_importance_plot.png")
    plt.savefig(feature_importance_path, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved at {feature_importance_path}")
    print(f"SHAP summary plot saved at {plot_path}")
    print(f"Evaluation metrics saved at {scores_file}")
    
    return shap_vals_to_plot, test_embeddings, feature_names, plot_title
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Set larger global font sizes for all plots
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })

    # Extract task name and embedding type from plot_title
    task_name = plot_title.split(" Prediction")[0]
    embedding_type = "Adapted" if "Adapted" in plot_title else "Original" if "Original" in plot_title else "Baseline"

    # Load embeddings and labels
    train_embeddings = pd.read_csv(f"data2/embeddings_avg/train_{embedding_path}_embeddings.csv").to_numpy()
    test_embeddings = pd.read_csv(f"data2/embeddings_avg/test_{embedding_path}_embeddings.csv").to_numpy()
    train_df = pd.read_csv(f"data2/train{sequence_path}.csv")
    test_df = pd.read_csv(f"data2/test{sequence_path}.csv")
    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()
    
    # Scale the data
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Train Gradient Boosting Classifier
    print(f"Training Gradient Boosting Classifier for {plot_title}...")
    gb = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(train_embeddings, train_labels_encoded)

    # Evaluate the model
    print(f"Evaluating Gradient Boosting Classifier for {plot_title}...")
    test_preds = gb.predict(test_embeddings)
    accuracy = accuracy_score(test_labels_encoded, test_preds)
    precision = precision_score(test_labels_encoded, test_preds)
    recall = recall_score(test_labels_encoded, test_preds)
    f1 = f1_score(test_labels_encoded, test_preds)
    test_probs = gb.predict_proba(test_embeddings)[:, 1]
    auroc = roc_auc_score(test_labels_encoded, test_probs)
    
    # Add results to our collection for summary table
    results_data.append({
        'Task': task_name,
        'Embedding Type': embedding_type,
        'ROC AUC': auroc,
        'Accuracy': accuracy,
        'F1 Score': f1
    })

    # Save scores to file
    scores_file = os.path.join(output_path, "scores.txt")
    with open(scores_file, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"AUROC: {auroc:.4f}\n")

    # SHAP Analysis
    print(f"Calculating SHAP values for {plot_title}...")
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(test_embeddings)
    
    # For HistGradientBoostingClassifier, shap_values might be directly usable
    # But if it's a list (for binary classification), we need to extract the second element
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals_to_plot = shap_values[1]
    else:
        shap_vals_to_plot = shap_values

    print(f"Plotting SHAP summary for {plot_title}...")
    if adapted:
        # Create feature names with provided task labels
        feature_names = []
        for i in range(test_embeddings.shape[1]):
            if i in task_labels:
                feature_names.append(task_labels[i])
            else:
                feature_names.append(f"feature_{i}")
    else:
        feature_names = [f"feature_{i}" for i in range(test_embeddings.shape[1])]

    plt.figure(figsize=(6, 16))  # Twice as high as wide
    shap.summary_plot(shap_vals_to_plot, features=test_embeddings, feature_names=feature_names,
                      max_display=10, show=False)
    plt.title(plot_title)

    # Save SHAP summary plot
    plot_path = os.path.join(output_path, "plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Calculate mean absolute SHAP values
    mean_shap_values = np.abs(shap_vals_to_plot).mean(axis=0)

    # Create a DataFrame for feature importance including the original indices
    importance_df = pd.DataFrame({
        'Index': range(test_embeddings.shape[1]),
        'Feature': feature_names,
        'Importance': mean_shap_values
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importance for the top 10 features
    top_features = importance_df.head(10)

    # Determine bar colors:
    # If adapted then features with an index less than CRAFTED_FEATURES are blue and the rest red.
    if adapted:
        bar_colors = ["#6C8EBF" if idx < CRAFTED_FEATURES else "#F2918C" for idx in top_features["Index"]]
    else:
        bar_colors = ["gray"] * len(top_features)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["Feature"], top_features["Importance"], color=bar_colors)
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title(f"SHAP Feature Importance\n({plot_title})")
    ax.invert_yaxis()  # Highest importance on top

    plt.tight_layout()
    feature_importance_path = os.path.join(output_path, "feature_importance_plot.png")
    plt.savefig(feature_importance_path, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved at {feature_importance_path}")
    print(f"SHAP summary plot saved at {plot_path}")
    print(f"Evaluation metrics saved at {scores_file}")
    
    # Return the shap values, feature names, and plot title for summary grid
    return shap_vals_to_plot, test_embeddings, feature_names, plot_title

def create_summary_grid(all_results):
    """Create a 3x3 grid of SHAP summary plots"""
    # Create a directory for the summary
    summary_dir = "output_gradient_boost"
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, (shap_vals, X, feat_names, title) in enumerate(all_results):
        # Get the current subplot
        row, col = i // 3, i % 3
        
        # Adjust title to be more concise
        task = title.split(" Prediction")[0]
        embedding = title.split("with ")[1]
        short_title = f"{task} - {embedding}"
        
        # Create SHAP summary for this subplot - save to a temporary file
        plt.figure(figsize=(8, 8))
        shap.summary_plot(
            shap_vals, 
            features=X, 
            feature_names=feat_names,
            max_display=10,
            show=False,
            title=None,
            alpha=0.8
        )
        plt.title(short_title, fontsize=18)
        
        # Save this individual plot
        temp_path = os.path.join(summary_dir, f"temp_plot_{i}.png")
        plt.savefig(temp_path, bbox_inches='tight')
        plt.close()
        
        # Now load this image into the main figure
        img = plt.imread(temp_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        
        # Clean up the temporary file
        os.remove(temp_path)
    
    # Save the grid plot
    plt.savefig(os.path.join(summary_dir, "shap_summary_grid.png"), bbox_inches='tight', dpi=300)
    print(f"Summary grid saved to {os.path.join(summary_dir, 'shap_summary_grid.png')}")
    plt.close()

def create_metrics_table(results_data):
    """Create a CSV table with all metrics"""
    df = pd.DataFrame(results_data)
    
    # Format the numeric columns
    for col in ['ROC AUC', 'Accuracy', 'F1 Score']:
        df[col] = df[col].map(lambda x: f"{x:.4f}")
    
    # Sort by task and embedding type
    df = df.sort_values(['Task', 'Embedding Type'])
    
    # Save to CSV
    csv_path = os.path.join("output_gradient_boost", "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics summary saved to {csv_path}")
    
    return df

if __name__ == "__main__":
    # Store results for final summary grid
    all_shap_results = []
    
    # AGG datasets
    result = create_plots('AGG_latent', "_sequences_AGG", "output_gradient_boost/Aggregation_adapted",
                "Aggregation Prediction with Adapted Embeddings", True, ["#F2918C", "#6C8EBF"])
    all_shap_results.append(result)
    
    result = create_plots('AGG_truncated_latent', "_sequences_AGG", "output_gradient_boost/Aggregation_baseline",
                "Aggregation Prediction with Baseline Embeddings", True, ["#F2918C", "#6C8EBF"])
    all_shap_results.append(result)

    result = create_plots('AGG_valid', "_sequences_AGG", "output_gradient_boost/Aggregation_original",
                "Aggregation Prediction with Original Embeddings", False, ["#B0B0B0", "#B0B0B0"])
    all_shap_results.append(result)
    
    result = create_plots('AGG_binary', "_sequences_AGG", "output_gradient_boost/Aggregation_baseline_binary",
                "Aggregation Prediction with Baseline Embeddings", True, ["#6C8EBF", "#6C8EBF"])
    all_shap_results.append(result)
    
    # EV datasets
    result = create_plots('EV_latent', "_sequences_EV", "output_gradient_boost/EV_adapted",
                "EV Prediction with Adapted Embeddings", True, ["#F2918C", "#6C8EBF"])
    all_shap_results.append(result)

    result = create_plots('EV_truncated_latent', "_sequences_EV", "output_gradient_boost/EV_baseline",
                "EV Prediction with Baseline Embeddings", True, ["#F2918C", "#6C8EBF"])
    all_shap_results.append(result)
    
    result = create_plots('EV_valid', "_sequences_EV", "output_gradient_boost/EV_original",
                "EV Prediction with Original Embeddings", False, ["#B0B0B0", "#B0B0B0"])
    all_shap_results.append(result)
    
    result = create_plots('EV_binary', "_sequences_EV", "output_gradient_boost/EV_baseline_binary",
                "EV Prediction with Baseline Embeddings", True, ["#6C8EBF", "#6C8EBF"])
    all_shap_results.append(result)
    
    # TM datasets
    result = create_plots('TM_latent', "_sequences_TM", "output_gradient_boost/TM_adapted",
                "TM Prediction with Adapted Embeddings", True, ["#F2918C", "#6C8EBF"])
    all_shap_results.append(result)

    result = create_plots('TM_truncated_latent', "_sequences_TM", "output_gradient_boost/TM_baseline",
                "TM Prediction with Baseline Embeddings", True, ["#F2918C", "#6C8EBF"])
    all_shap_results.append(result)
    
    result = create_plots('TM_valid', "_sequences_TM", "output_gradient_boost/TM_original",
                "TM Prediction with Original Embeddings", False, ["#B0B0B0", "#B0B0B0"])
    all_shap_results.append(result)
    
    result = create_plots('TM_binary', "_sequences_TM", "output_gradient_boost/TM_baseline_binary",
                "TM Prediction with Baseline Embeddings", True, ["#6C8EBF", "#6C8EBF"])
    all_shap_results.append(result)
    
    # Create 3x3 summary grid of all SHAP plots
    # create_summary_grid(all_shap_results)
    
    # Create metrics summary table
    create_metrics_table(results_data)