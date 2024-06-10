import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

def train_xgboost_model():
    df = pd.read_csv('heart.csv')

    df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

    expected_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    numerical_attributes = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'FastingBS']
    scaler = MinMaxScaler()
    df[numerical_attributes] = scaler.fit_transform(df[numerical_attributes])

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgboost_params = {
        'objective': 'binary:logistic',
        'colsample_bytree': 0.6,
        'gamma': 0.4,
        'learning_rate': 0.2,
        'max_depth': 4,
        'min_child_weight': 5,
        'subsample': 0.9,
        'n_estimators': 100
    }

    xgboost_model = XGBClassifier(**xgboost_params)
    trained_model = xgboost_model.fit(X_train, y_train)
    return trained_model
