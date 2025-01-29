from matplotlib import cm, patches
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

CRAFTED_FEATURES = 34

task_labels = {
    0: 'ASA', 
    1: 'SS8_H', 2: 'SS8_E', 3: 'SS8_G', 4: 'SS8_I', 5: 'SS8_B', 6: 'SS8_T', 7: 'SS8_S', 8: 'SS8_-',
    9: 'SS3_H', 10: 'SS3_E', 11: 'SS3_C',
    12: 'A', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'K', 
    21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 
    27: 'S', 28: 'T', 29: 'V', 30: 'W', 31: 'Y', 
    32: 'GRAVY', 
    33: "AROM"
}


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import os

def create_plots(embedding_path, sequence_path, output_path, plot_title, adapted, custom_gradient):
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load embeddings and labels
    train_embeddings = pd.read_csv(f"data/embeddings_avg/train_{embedding_path}_embeddings.csv").to_numpy()
    test_embeddings = pd.read_csv(f"data/embeddings_avg/test_{embedding_path}_embeddings.csv").to_numpy()

    train_df = pd.read_csv(f"data/train{sequence_path}.csv")
    test_df = pd.read_csv(f"data/test{sequence_path}.csv")

    train_labels = train_df["label"].tolist()
    test_labels = test_df["label"].tolist()
    
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Train XGBoost Classifier
    print("Training XGBoost Classifier...")
    xgb = XGBClassifier(
        n_estimators=50,
        max_depth=7,
        learning_rate=0.1,
        random_state=42
    )
    xgb.fit(train_embeddings, train_labels_encoded)

    # Evaluate the model
    print("Evaluating XGBoost Classifier...")
    test_preds = xgb.predict(test_embeddings)

    accuracy = accuracy_score(test_labels_encoded, test_preds)
    precision = precision_score(test_labels_encoded, test_preds)
    recall = recall_score(test_labels_encoded, test_preds)
    f1 = f1_score(test_labels_encoded, test_preds)
    test_probs = xgb.predict_proba(test_embeddings)[:, 1]
    auroc = roc_auc_score(test_labels_encoded, test_probs)

    # Save scores to file
    scores_file = os.path.join(output_path, "scores.txt")
    with open(scores_file, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"AUROC: {auroc:.4f}\n")

    # SHAP Analysis
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(test_embeddings)

    print("Plotting SHAP summary...")
    if adapted:
        feature_names = [task_labels[i] if i in task_labels and task_labels[i] is not None else f"Feature_{i}" for i in range(test_embeddings.shape[1])]
    else:
        feature_names = [f"Feature_{i}" for i in range(test_embeddings.shape[1])]

    plt.figure(figsize=(6, 16))  # Twice as high as wide
    shap.summary_plot(shap_values, features=test_embeddings, feature_names=feature_names, max_display=10, show=False, title=plot_title)

    # Save plot
    plot_path = os.path.join(output_path, "plot.png")
    plt.savefig(plot_path)
    plt.close()

    # Calculate mean absolute SHAP values
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame for feature names and importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_shap_values
    }).sort_values(by="Importance", ascending=False)

    # Define a custom colormap from DAE8FC (light blue) to F8CECC (light red)
    custom_colormap = mcolors.LinearSegmentedColormap.from_list(
        "custom_gradient", custom_gradient
    )

    fig, ax = plt.subplots(figsize=(10, 7))  # Increased figure height for more room

    # Adjust the spacing between bars
    bar_height = 0.6  # Height of each bar, reduced to leave space between bars

    # Loop through top 10 features and create gradient bars
    for idx, (feature, importance) in enumerate(importance_df.iloc[:10].values):
        # Create a horizontal gradient
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.repeat(gradient, 10, axis=0)  # Repeat to create a "bar" effect

        # Adjust the gradient position to fit fully within the bar
        bar_top = idx + bar_height / 2
        bar_bottom = idx - bar_height / 2

        # Plot the gradient as an image, limited to the SHAP value's length
        ax.imshow(
            gradient,
            aspect='auto',
            extent=(0, importance, bar_bottom, bar_top),  # Properly position the gradient
            cmap=custom_colormap,
            origin='lower'
        )

    # Add feature names and labels
    ax.set_yticks(range(10))
    ax.set_yticklabels(importance_df["Feature"][:10])
    ax.set_xlim(0, importance_df["Importance"].max() * 1.1)
    ax.set_ylim(-0.5, 9.5 + (1 - bar_height))
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title(f"SHAP feature importance\n({plot_title})")
    ax.invert_yaxis()

    plt.rc('font', size=18)  # Base font size for the plot
    plt.rc('axes', titlesize=22)  # Title font size
    plt.rc('axes', labelsize=20)  # X and Y labels font size
    plt.rc('xtick', labelsize=20)  # X tick labels font size
    plt.rc('ytick', labelsize=20)  # Y tick labels font size
    plt.rc('legend', fontsize=20)  # Legend font size
    plt.rc('figure', titlesize=20)

    plt.subplots_adjust(left=0.20)  # Increase left margin

    # Save the plot
    feature_importance_path = os.path.join(output_path, "feature_importance_plot.png")
    plt.savefig(feature_importance_path)
    plt.close()

    print(f"Feature importance plot saved at {feature_importance_path}")
    print(f"SHAP summary plot saved at {plot_path}")
    print(f"Evaluation metrics saved at {scores_file}")


if __name__ == "__main__":
    create_plots('EV_latent', "_sequences_EV", "output/EV_adapted", "EV Prediction with Adapted Embeddings", True, ["#F2918C", "#6C8EBF"])
    create_plots('EV_valid', "_sequences_EV", "output/EV_original", "EV Prediction with Original Embeddings", False, ["#B0B0B0", "#B0B0B0"])
    create_plots('EV_binary', "_sequences_EV", "output/EV_baseline", "EV Prediction with Baseline Embeddings", True, ["#6C8EBF", "#6C8EBF"])
    # create_plots('EV_truncated_latent', "_sequences_EV", "output/EV_adapted_only_crafted", "SHAP for EV Prediction with Truncated Adapted Embeddings", True)
    create_plots('latent', "", "output/Aggregation_adapted", "Aggregation Prediction with Adapted Embeddings", True, ["#F2918C", "#6C8EBF"])
    create_plots('valid', "", "output/Aggregation_original", "Aggregation Prediction with Original Embeddings", False, ["#B0B0B0", "#B0B0B0"])
    create_plots('binary', "", "output/Aggregation_baseline", "Aggregation Prediction with Baseline Embeddings", True, ["#6C8EBF", "#6C8EBF"])
    # create_plots('truncated_latent', "", "output/Aggregation_adapted_only_crafted", "SHAP for Aggregation Prediction with Truncated Adapted Embeddings", True)
    # create_plots('tm_2_latent', "_sequences", "output/TM_2_adapted", "TM Prediction with Adapted Embeddings", True, ["#F2918C", "#6C8EBF"])
    # create_plots('tm_2_valid', "_sequences", "output/TM_2_original", "TM Prediction with Original Embeddings", False, ["#B0B0B0", "#B0B0B0"])
    # create_plots('tm_2_binary', "_sequences", "output/TM_2_baseline", "TM Prediction with Baseline Embeddings", True, ["#6C8EBF", "#6C8EBF"])
    # create_plots('TM_truncated_latent', "_sequences", "output/TM_adapted_only_crafted", "SHAP for TM Prediction with Truncated Adapted Embeddings", True)