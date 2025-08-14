# ---------------------------
# 8) OPTIONAL: Per-transaction explanations with SHAP (recommended for better notes)
# ---------------------------
# If you want per-row feature contributions (which features increased the fraud probability for *this* transaction),
# use SHAP. Install shap: pip install shap
#
# Example (uncomment to run if shap installed):
#
# import shap
# explainer = shap.TreeExplainer(rf_sm)
# # Use a small set for performance (or use the whole test set)
# shap_values = explainer.shap_values(X_test_reset, check_additivity=False)
# # shap_values[1] is for the positive class (fraud)
# top_n = 5
# per_row_top_feats = []
# for i in range(X_test_reset.shape[0]):
#     row_shap = pd.Series(shap_values[1][i], index=X_test_reset.columns)
#     row_shap_abs = row_shap.abs().sort_values(ascending=False)
#     top_feats = list(row_shap_abs.head(top_n).index)
#     top_contributions = ", ".join([f"{f}({row_shap[f]:+.3f})" for f in top_feats])
#     per_row_top_feats.append(top_contributions)

# audit_df["Top_SHAP_Features"] = per_row_top_feats
# print("\nExample with SHAP top contributing features:")
# print(audit_df[audit_df["Prediction"]==1].head(5)[["idx","Probability","Actual","Top_SHAP_Features"]].to_string(index=False))

# # You can then incorporate Top_SHAP_Features into the Audit_Note for a much more meaningful explanation.
