import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices, build_design_matrices
import seaborn as sns
from sklearn.linear_model import LogisticRegression


class HeartDiseaseModel:
    """
    A model for heart disease prediction using spline features and logistic regression.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The heart disease dataset used for fitting the model and determining knot positions.
    feature_types : dict
        Dictionary mapping feature names to their types ('numerical' or 'categorical').
    spine_df : int, default=4
        Degrees of freedom for numerical features.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, data, feature_types, spline_df=4, random_state=None):
        self._data = data
        self._feature_types = feature_types
        self._validate_feature_types()
        self.spline_df = spline_df
        self.model = LogisticRegression(
            penalty=None,
            fit_intercept=False, # intercept is included in design matrix
            random_state=random_state,
        )
        self.X = None
        self.y = None
        self.covariance_matrix = None
        self._is_fitted = False

    def fit(self, target_col='chd'):
        """Fit the model to the data."""
        formula = f"{target_col} ~ {self._make_dmatrix_formula()}"
        self.y, self.X = dmatrices(formula, self._data)
        self.model.fit(self.X, self.y)
        self._is_fitted = True
        self.covariance_matrix = self._estimate_covariance_matrix()
        return self
    
    def predict_proba(self, new_data):
        """Predict probability of heart disease."""
        self._check_is_fitted()
        X_new = build_design_matrices([self.X.design_info], new_data)[0]
        return self.model.predict_proba(X_new)[:, 1]
    
    def feature_effect(self, feature, x_vals=None, return_std_err=False):
        """Calculate the effect of varying a single feature on the linear predictor."""
        self._check_is_fitted()

        if feature not in self._feature_types:
            raise ValueError(f"Feature {feature} not in model.")
        
        # Get the design info for just this feature
        feature_formula = self._make_feature_formula(feature, self._feature_types[feature])
        subdesign_info = self.X.design_info.subset(feature_formula + "-1")

        if x_vals is None:
            if self._feature_types[feature] == "categorical":
                # Get categories directly from the factor info
                factor_info = next(iter(subdesign_info.factor_infos.values()))
                x_vals = factor_info.categories
            else:
                x_vals = np.linspace(self._data[feature].min(), self._data[feature].max(), 100)
        
        # Build design matrix for this feature
        X_vals = build_design_matrices(
            [subdesign_info],
            {feature: x_vals}
        )[0]
        
        # Compute the linear predictor values (before logistic transformation)
        feature_slice = self.X.design_info.slice(feature_formula)
        y_vals = X_vals @ self.model.coef_[0, feature_slice]

        std_err = None
        if return_std_err:
            std_var = np.diagonal(
                X_vals @ self.covariance_matrix[feature_slice, feature_slice] @ X_vals.T
            )
            std_err = np.sqrt(std_var)

        return x_vals, y_vals, std_err
        
    def plot_all_feature_effects(self, plot_std_err_bands=True, figsize=(12, 10), n_cols=2):
        """Plot effects of all features in a grid layout."""
        self._check_is_fitted()
        
        features = list(self._feature_types.keys())
        n_features = len(features)
        
        n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes for easier indexing
        if n_features > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each feature
        for i, feature in enumerate(features):
            if self._feature_types[feature] == "numerical":
                self.plot_numerical_feature_effect(feature, plot_std_err_bands=plot_std_err_bands, ax=axes[i])
            elif self._feature_types[feature] == "categorical":
                self.plot_categorical_feature_effect(feature, plot_std_err_bands=plot_std_err_bands, ax=axes[i])
        
        # Hide unused axes
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle("South African Heart Disease Feature Effects")
        fig.tight_layout()
        
        return fig, axes
    
    def plot_numerical_feature_effect(self, feature, x_vals=None, plot_std_err_bands=True, color=None, ax=None):
        """Plot the effect of a single numerical feature."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        if color is None:
            color = sns.color_palette()[0]
            
        x_vals, y_vals, std_err = self.feature_effect(feature, x_vals, return_std_err=plot_std_err_bands)
        
        ax.plot(x_vals, y_vals, color=color)

        sns.rugplot(data=self._data, x=feature, ax=ax, color=color, alpha=0.5, height=0.05)

        if plot_std_err_bands:
            ax.fill_between(x_vals, y_vals - 2 * std_err, y_vals + 2 * std_err, color=color, alpha=0.2)

        ax.set(
            xlabel=r"$\mathtt{" + feature + "}$",
            ylabel=r"$\hat{f}~(\mathtt{" + feature + "})$",
        )
        
        return ax
    
    def plot_categorical_feature_effect(self, feature, plot_std_err_bands=True, color=None, ax=None):
        """Plot the effect of a single categorical feature on the linear predictor."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        
        if color is None:
            color = sns.color_palette()[0]
        
        categories, effects, std_err = self.feature_effect(feature, return_std_err=plot_std_err_bands)
        
        # plot effects
        positions = np.arange(len(categories))
        ax.hlines(effects, xmin=positions-0.4, xmax=positions+0.4, colors=color, linewidth=2, capstyle='round')
        
        # Add a marker at each line
        for i, (pos, effect) in enumerate(zip(positions, effects)):
            ax.plot(pos, effect, 'o', markersize=4, color=color)

        # Add shaded error bands
        if plot_std_err_bands:
            for i, (pos, effect, err) in enumerate(zip(positions, effects, std_err)):
                x_band = [pos - 0.25, pos + 0.25]
                lower_bound = effect - 2 * err
                upper_bound = effect + 2 * err
                
                ax.fill_between(x_band, [lower_bound, lower_bound], [upper_bound, upper_bound], color=color, alpha=0.2)
        
        if plot_std_err_bands is False:
            # Add vertical padding
            y_range = max(effects) - min(effects)
            padding = 0.25 * y_range if y_range > 0 else 0.5
            ax.set_ylim([min(effects) - padding, max(effects) + padding])
    
        ax.set(
            xticks=positions,
            xticklabels=categories,
            xlabel=r"$\mathtt{" + feature + "}$",
            ylabel=r"$\hat{f}~(\mathtt{" + feature + "})$",
        )
        
        return ax

    def _validate_feature_types(self):
        """Validate that all features specified in feature_types exist in the data."""
        missing_features = [feature for feature in self._feature_types if feature not in self._data.columns]
        if missing_features:
            raise ValueError(f"The following features are not present in the data: {missing_features}")
        
    def _make_feature_formula(self, feature, feature_type):
        """Create the formula part for a single feature."""
        if feature_type == "numerical":
            # patsy calculates knots based on quantiles after removing duplicate x-values
            # to replicate the textbook we need to calculuate them manually
            # this is actually much worse - we get errors if two knots are the same (or equal to lower/upper bounds)
            n_inner_knots = self.spline_df - 1
            inner_knots_q = np.linspace(0, 100, n_inner_knots + 2)[1:-1]
            inner_knots = list(np.percentile(self._data[feature], inner_knots_q))

            return f"cr({feature}, df={self.spline_df}, knots={inner_knots}, constraints='center')"
        elif feature_type == "categorical":
            return feature
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def _make_dmatrix_formula(self):
        """Create the complete formula for the design matrix."""
        feature_formulas = [
            self._make_feature_formula(feature, feature_type)
            for feature, feature_type in self._feature_types.items()
        ]
        return " + ".join(feature_formulas) + "-1"
    
    def _diagonal_weight_matrix(self):
        """Array with values p(x_i)(1 - p(x_i)) on the diagonal."""
        weights = self.model.predict_proba(self.X).prod(axis=1)
        return np.diag(weights)

    def _estimate_covariance_matrix(self):
        """
        Estimate the covariance matrix of the model parameters.
        
        For logistic regression, the covariance matrix is calculated as:
        cov(β) = (X^T W X)^(-1)
        where W is a diagonal matrix of weights p_i(1-p_i)
        """
        W = self._diagonal_weight_matrix()
        X = self.X
        return np.linalg.inv(X.T @ W @ X)
    
    def _check_is_fitted(self):
        """Verify the model is fitted before operations that require it."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before this operation.")
