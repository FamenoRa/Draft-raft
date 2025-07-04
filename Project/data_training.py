# for training the congestion model

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_congestion_model(df: pd.DataFrame) -> RandomForestClassifier:
    """Train model to predict congestion level."""
    features = df[['hour', 'vehicle_count']]
    if 'congestion_level' in df.columns:
        labels = df['congestion_level']
    else:
        labels = pd.cut(df['vehicle_count'], bins=3, labels=[0,1,2])
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model
