import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from scipy.special import expit  # logistic function


def preprocess_penalties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer Round_Number and Penalty_Basket from Penalty_Number.

    Kicks alternate between teams, so Round_Number = ceil(Penalty_Number / 2).
    Penalty_Basket groups Round_Number into 6 categorical bins (1-indexed for Stan):
      1: Round 1
      2: Round 2
      3: Round 3
      4: Rounds 4 & 5
      5: Rounds 6-10
      6: Rounds 11+
    """
    out = df.copy()
    out["Round_Number"] = np.ceil(out["Penalty_Number"] / 2).astype(int)

    out["Penalty_Basket"] = np.select(
        [
            out["Round_Number"] == 1,
            out["Round_Number"] == 2,
            out["Round_Number"] == 3,
            (out["Round_Number"] == 4) | (out["Round_Number"] == 5),
            (out["Round_Number"] >= 6) & (out["Round_Number"] <= 10),
            out["Round_Number"] >= 11
        ],
        [1, 2, 3, 4, 5, 6],
        default=6
    ).astype(int)
    return out


def calc_metrics(y_true, y_prob, y_pred):
    return {
        'Log-Loss': log_loss(y_true, y_prob),
        'ROC AUC': roc_auc_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Brier Score': brier_score_loss(y_true, y_prob)
    }


def predict_base(row, draws):
    eta = (draws[f"beta_basket[{row['Penalty_Basket']}]"]
           + draws["beta_elim"] * row["Elimination"]
           + draws["beta_left"] * row["is_left"])
    return expit(eta).mean()


def predict_pos(row, draws):
    eta = (draws[f"alpha_pos[{row['Position_ID']}]"]
           + draws[f"beta_basket[{row['Penalty_Basket']}]"]
           + draws["beta_elim"] * row["Elimination"]
           + draws["beta_left"] * row["is_left"])
    return expit(eta).mean()
