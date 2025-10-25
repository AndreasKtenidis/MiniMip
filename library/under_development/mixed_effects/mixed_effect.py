import pandas as pd
import numpy as np

def compute_SXX_for_center(X_j:np.array,Y_j:np.array, sigma2, sigma_u2):
    """Compute S_XX for a single center"""
    n_j = X_j.shape[0]
    I_nj = np.eye(n_j)
    ones = np.ones((n_j, 1))

    alpha_j = sigma_u2 / (sigma2 * (sigma2 + n_j * sigma_u2))
    V_inv = (1 / sigma2) * I_nj - alpha_j * ones @ ones.T

    S_XX = X_j.T @ V_inv @ X_j
    S_YY = Y_j.T @ V_inv @ Y_j
    print(S_XX)
    return V_inv, S_XX, S_YY

def mixed_effect(patients,*,covariates,center,outcome) :
    # Group by center and iterate through each center
    centers_grouped = patients.groupby(center)
    print(centers_grouped)
    print("=== Method 1: Basic Loop ===")
    sigma2 = patients[outcome].var()   # Assume variance within the population
    for center_id, center_data in centers_grouped:
        x_j=center_data[covariates]
        y_j = center_data[outcome]
        sigma2_j=center_data[outcome].var()
        compute_SXX_for_center(x_j.values,y_j.values, sigma2, sigma2_j)

    return 0

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(123)
    # Parameters
    n_patients = 100
    n_centers = 8  # Adjust as needed
    # Generate random data
    patients = pd.DataFrame({
        'center': np.random.choice(['A', 'B', 'C'], n_patients),
        'T': np.random.choice([0, 1], n_patients),
        'age': np.random.normal(45, 10, n_patients),
        'baseline_pain': np.random.normal(5, 2, n_patients),
        'Y': np.random.normal(0, 1, n_patients)
    })
    v_inv, s_xx,s_yy = mixed_effect(patients,covariates=['T','age','baseline_pain'],center='center',outcome='Y')
