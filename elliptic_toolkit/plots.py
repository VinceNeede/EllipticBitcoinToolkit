import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import PrecisionRecallDisplay, average_precision_score


def _get_marginals_ticks(x_labels, N=10):
    """
    Helper for plot_marginals: reduces the number of x-ticks for readability.
    """
    x = np.array(range(len(x_labels))
                 )  # convert to numpy array for fancy indexing
    x_labels = np.array(x_labels)  # convert to numpy array for fancy indexing

    if len(x_labels) > N:
        step = max(1, len(x_labels) // N)
        shown_idx = list(range(0, len(x_labels), step))

        return x[shown_idx], x_labels[shown_idx]
    return x, x_labels


def plot_marginals(cv_results, max_ticks=10):
    """
    For each hyperparameter in ``cv_results``, plot the marginal mean and standard deviation (error bar) of test scores.

    The marginal mean/std for each hyperparameter value is computed by averaging across all other hyperparameters
    the mean/std across the cv folds
    (i.e., by computing the average of the ``mean_test_score`` and ``std_test_score`` columns).

    Parameters
    ----------
    cv_results : dict
        The ``cv_results_`` attribute from a scikit-learn search object.
    max_ticks : int, default=10
        Maximum number of x-ticks to show on the x-axis for readability.

    Returns
    -------
    figs : dict
        Dictionary mapping parameter names to ``matplotlib.figure.Figure`` objects.
    """
    results = pd.DataFrame(cv_results)
    param_names = [col for col in results.columns if col.startswith('param_')]
    figs = dict()
    for param in param_names:
        fig, ax = plt.subplots()
        # Group by the parameter and compute mean test score
        marginals = results.groupby(param, dropna=False)[
            'mean_test_score'].mean()
        marginals_std = results.groupby(param, dropna=False)[
            'std_test_score'].mean()
        x_labels = [f"{x:.2g}" if isinstance(
            x, float) else str(x) for x in marginals.index]
        ax.errorbar(x_labels, marginals, yerr=marginals_std, fmt='-o')
        ax.set_title(f'Marginal mean test score for {param}')
        ax.set_ylabel('Mean test score')
        xticks, xticks_labels = _get_marginals_ticks(x_labels, max_ticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels, rotation=45)
        figs[param] = fig
    return figs


def plot_evals(est, X_test, y_test, y_train, *, time_steps_test=None):
    """
    Generate two evaluation plots for a classifier:
    1. Precision-Recall curve on the test set.
    2. Rolling/cumulative AP and illicit rate by time step.

    Parameters
    ----------
    est : classifier
        Trained classifier with predict_proba method.
    X_test : pd.DataFrame, array-like
        Test features. Must contain a 'time' column unless time_steps_test is provided.
    y_test : numpy.ndarray
        Test labels (binary).
    y_train : numpy.ndarray
        Training labels (binary), used for reference illicit rate.
    time_steps_test : numpy.ndarray, optional
        Time step values for test set. If None, will use X_test['time'].

    Returns
    -------
    pr_fig : matplotlib.figure.Figure
        Figure for the precision-recall curve.
    temporal_fig : matplotlib.figure.Figure
        Figure for the rolling/cumulative AP and illicit rate by time step.

    Notes
    -----
    This function assumes arrays to be numpy ndarrays. ``X_test`` is allowed to be a torch.Tensor
    but est.predict_proba must return numpy arrays.
    """
    y_pred_proba = est.predict_proba(X_test)[:, 1]

    pr_fig, pr_ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(
        y_test, y_pred_proba, plot_chance_level=True, ax=pr_ax)

    # Get time steps for test data
    if time_steps_test is None:
        if hasattr(X_test, 'time'):
            time_steps_test = X_test['time'].values
        else:
            raise ValueError(
                'either pass time_steps_test esplicitly or X_test must have column ``time``')

    # Create results DataFrame
    results_df = pd.DataFrame({
        'time': time_steps_test,
        'actual': y_test,
        'pred_proba': y_pred_proba,
    })

    # Get unique time steps in ascending order
    unique_times = sorted(results_df['time'].unique())

    # Prepare data structures for rolling analysis
    rolling_metrics = []

    # For each cutoff point, calculate metrics on all data up to and including
    # that time step
    for i, cutoff_time in enumerate(unique_times):
        # Select data up to and including the current time step
        current_data = results_df[results_df['time'] <= cutoff_time]

        current_ap = average_precision_score(
            current_data['actual'], current_data['pred_proba'])
        current_illicit_rate = np.mean(current_data['actual'] == 1)

        rolling_metrics.append({
            'cutoff_time': cutoff_time,
            'ap': current_ap,
            'illicit_rate': current_illicit_rate,
            'sample_size': len(current_data),
            'illicit_count': sum(current_data['actual'] == 1)
        })

    # Convert to DataFrame
    rolling_df = pd.DataFrame(rolling_metrics)

    # Calculate training set illicit rate for reference
    train_illicit_rate = np.mean(y_train == 1)

    # Create figure with two y-axes
    temporal_fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot AP on primary y-axis
    ax1.plot(rolling_df['cutoff_time'], rolling_df['ap'], 'b-o', linewidth=2,
             label='Rolling AP')
    ax1.set_xlabel('Time Step Cutoff', fontsize=12)
    ax1.set_ylabel('AP', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    # Create a secondary y-axis for illicit rates
    ax2 = ax1.twinx()
    ax2.plot(
        rolling_df['cutoff_time'],
        rolling_df['illicit_rate'],
        'r-^',
        linewidth=2,
        label='Rolling Illicit Rate')
    ax2.axhline(y=train_illicit_rate, color='r', linestyle='--', alpha=0.7,
                label=f'Train Illicit Rate: {train_illicit_rate:.3f}')
    ax2.set_ylabel('Illicit Rate', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax1.set_title('Rolling/Cumulative Performance by Time Step', fontsize=15)
    return pr_fig, temporal_fig
