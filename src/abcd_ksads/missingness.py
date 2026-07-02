"""Quantify the 555-as-0 missingness error for a single diagnosis variable.

Treating not-administered (555) cells as negatives inflates the denominator and
deflates prevalence. This computes the correct prevalence (over the assessed
denominator) versus the erroneous one (over all person-waves), and the fold by which
the error deflates it. The ``10c_aux_missingness_audit.py`` script wires it to the CSV.
"""


def missingness_error(d):
    """From resolved rows for one variable, return the deflation statistics.

    ``d`` has a ``resolved`` column over {positive, administered_negative,
    not_administered, no_record}; every row is one person-wave in the release."""
    n_pos = int((d.resolved == "positive").sum())
    n_assessed = int(d.resolved.isin(["positive", "administered_negative"]).sum())
    n_all = int(len(d))  # every person-wave with a row (incl. 555 and no_record)
    n_not_admin = int((d.resolved == "not_administered").sum())

    correct = 100 * n_pos / n_assessed
    error = 100 * n_pos / n_all
    fold = correct / error
    return {
        "n_positive": n_pos,
        "n_administered": n_assessed,
        "n_all_personwaves": n_all,
        "prevalence_correct_pct": round(correct, 3),
        "prevalence_error_pct": round(error, 3),
        "fold_deflation": round(fold, 2),
        "fabricated_personwaves": n_not_admin,
    }
