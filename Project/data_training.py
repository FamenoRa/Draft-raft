# for training the congestion model

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_congestion_model(df: pd.DataFrame) -> RandomForestClassifier:
    """Train RandomForest on hour and vehicle_count to predict congestion."""
    X = df[['hour','vehicle_count']]
    if 'congestion_level' in df.columns:
        y = df['congestion_level']
    else:
        y = pd.cut(df['vehicle_count'], bins=3, labels=[0,1,2])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model