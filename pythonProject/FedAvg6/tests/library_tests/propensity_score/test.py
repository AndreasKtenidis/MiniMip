import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors



def propensity_score_non_fd(df,*,features, treatment):


    # Estimate propensity scores
    X = df[features]
    y = df[treatment]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    # Split treatment/control
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    # Match using Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])

    matched_control = control.iloc[indices.flatten()]
    matched_df = pd.concat([treated.reset_index(drop=True), matched_control.reset_index(drop=True)], axis=0)
    return matched_control,matched_df

