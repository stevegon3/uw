def stepwise_selection_insurance(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    if initial_list is None:
        initial_list = []
    included = list(initial_list)
    while True:
        changed = False
        # Forward
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for col in excluded:
            X_model = sm.add_constant(X[included + [col]])
            model_insurance = sm.OLS(y, X_model).fit()
            new_pval[col] = model_insurance.pvalues[col]
        best_pval = new_pval.min() if not new_pval.empty else None
        if best_pval is not None and best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print("Add ", best_feature, "with p=", best_pval)
        # Backward
        if included:
            X_model = sm.add_constant(X[included])
            model_insurance = sm.OLS(y, X_model).fit()
            pvalues = model_insurance.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    print("Drop", worst_feature, "with p", worst_pval)
        if not changed:
            break
    return included