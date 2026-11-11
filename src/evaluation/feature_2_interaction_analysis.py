import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.inspection import partial_dependence


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "transactions.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "rf_threshold_model.pkl"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")

OUTPUT_PATH = os.path.join(
    REPORT_DIR,
    "feature_2_interaction_results.csv"
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TARGET_TIMESTEP = 42
GRID_RESOLUTION = 20

FEATURE_PAIRS = [
    ("feature_2", "feature_55"),
    ("feature_2", "feature_53"),
    ("feature_2", "feature_54"),
    ("feature_2", "feature_142"),
    ("feature_2", "feature_144")
]


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading dataset and model...")

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    feature_cols = [
        f"feature_{i}" for i in range(1, 166)
    ]

    required_cols = [
        "time_step",
        "class",
        *feature_cols
    ]

    dtype_map = {
        "time_step": "int8",
        "class": "category"
    }

    dtype_map.update({
        col: "float32"
        for col in feature_cols
    })

    df = pd.read_csv(
        DATA_PATH,
        usecols=required_cols,
        dtype=dtype_map
    )

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # FILTER TIMESTEP 42
    # --------------------------------------------------

    df = df[
        (df["time_step"] == TARGET_TIMESTEP) &
        (
            df["class"].astype(str).isin(
                ["licit", "illicit"]
            )
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No labelled transactions found for timestep 42."
        )

    df["target"] = (
        df["class"].astype(str) == "illicit"
    ).astype(int)

    X = df[feature_cols].copy()

    y = df["target"].to_numpy()

    print("\nDATASET")
    print("Transactions:", len(X))
    print("Licit:", int((y == 0).sum()))
    print("Illicit:", int((y == 1).sum()))

    # --------------------------------------------------
    # BASELINE
    # --------------------------------------------------

    baseline_prob = model.predict_proba(X)[:, 1]

    print(
        "\nBaseline mean illicit probability:",
        baseline_prob.mean()
    )

    # Keep DataFrame to preserve feature names
    X_array = X.astype(np.float64)

    # --------------------------------------------------
    # FEATURE INTERACTION ANALYSIS
    # --------------------------------------------------

    results = []

    for feature_a, feature_b in FEATURE_PAIRS:

        print("\n" + "=" * 55)

        print(
            f"ANALYSING {feature_a} x {feature_b}"
        )

        feature_a_idx = X.columns.get_loc(feature_a)
        feature_b_idx = X.columns.get_loc(feature_b)

        pd_result = partial_dependence(
            model,
            X_array,
            features=[(feature_a_idx, feature_b_idx)],
            kind="average",
            method="brute",
            response_method="predict_proba",
            grid_resolution=GRID_RESOLUTION,
            percentiles=(0.05, 0.95)
        )

        grid_a, grid_b = pd_result["grid_values"]

        # Binary classifier: select the available output surface
        average_values = pd_result["average"]

        if average_values.shape[0] == 1:
            surface = average_values[0]
        else:
            surface = average_values[1]

        # --------------------------------------------------
        # APPROXIMATE INTERACTION STRENGTH
        # --------------------------------------------------

        row_effect = surface.mean(
            axis=1,
            keepdims=True
        )

        column_effect = surface.mean(
            axis=0,
            keepdims=True
        )

        grand_mean = surface.mean()

        interaction_surface = (
            surface
            - row_effect
            - column_effect
            + grand_mean
        )

        interaction_rms = np.sqrt(
            np.mean(interaction_surface ** 2)
        )

        interaction_max = np.max(
            np.abs(interaction_surface)
        )

        results.append({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "interaction_rms": interaction_rms,
            "interaction_max_abs": interaction_max,
            "mean_predicted_probability": surface.mean()
        })

        print(
            "Mean predicted probability:",
            round(float(surface.mean()), 6)
        )

        print(
            "Interaction RMS:",
            round(float(interaction_rms), 6)
        )

        print(
            "Maximum interaction deviation:",
            round(float(interaction_max), 6)
        )

        # --------------------------------------------------
        # SAVE INTERACTION SURFACE
        # --------------------------------------------------

        rows = []

        for i, value_a in enumerate(grid_a):

            for j, value_b in enumerate(grid_b):

                rows.append({
                    "feature_a": feature_a,
                    "feature_a_value": value_a,
                    "feature_b": feature_b,
                    "feature_b_value": value_b,
                    "predicted_probability": surface[i, j],
                    "interaction_component":
                        interaction_surface[i, j]
                })

        surface_df = pd.DataFrame(rows)

        surface_path = os.path.join(
            REPORT_DIR,
            f"{feature_a}_{feature_b}_surface.csv"
        )

        surface_df.to_csv(
            surface_path,
            index=False
        )

        # --------------------------------------------------
        # VISUALIZATION
        # --------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 7))

        contour = ax.contourf(
            grid_a,
            grid_b,
            surface.T,
            levels=20
        )

        fig.colorbar(
            contour,
            ax=ax,
            label="Predicted Probability"
        )

        ax.set_xlabel(feature_a)
        ax.set_ylabel(feature_b)

        ax.set_title(
            f"Partial Dependence: {feature_a} x {feature_b}"
        )

        plt.tight_layout()

        figure_path = os.path.join(
            FIGURE_DIR,
            f"{feature_a}_{feature_b}_interaction.png"
        )

        plt.savefig(
            figure_path,
            dpi=300
        )

        plt.close()

        print("Surface saved:", surface_path)
        print("Figure saved:", figure_path)

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "interaction_rms",
        ascending=False
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 55)

    print("FEATURE INTERACTION SUMMARY")

    print(results_df.to_string(index=False))

    print("\nINVESTIGATION COMPLETED")

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()