from flask import Flask, request, render_template
from xgboost_model import train_xgboost_model
import pandas as pd

app = Flask(__name__)


sex_mapping = {'Male': 1, 'Female': 0}
chest_pain_mapping = {'ATA': 'ChestPainType_ATA', 'NAP': 'ChestPainType_NAP', 'TA': 'ChestPainType_TA'}
ecg_mapping = {'Normal': 'RestingECG_Normal', 'ST': 'RestingECG_ST'}
exercise_angina_mapping = {'Yes': 'ExerciseAngina_Y', 'No': 'ExerciseAngina_N'}
st_slope_mapping = {'Flat': 'ST_Slope_Flat', 'Up': 'ST_Slope_Up'}

def transform_input_data(input_data):
    for key, value in chest_pain_mapping.items():
        input_data[value] = 1 if input_data['chest_pain_type'].iloc[0] == key else 0
    
    for key, value in ecg_mapping.items():
        input_data[value] = 1 if input_data['ecg'].iloc[0] == key else 0
    
    for key, value in exercise_angina_mapping.items():
        input_data[value] = 1 if input_data['exercise_angina'].iloc[0] == key else 0
    
    for key, value in st_slope_mapping.items():
        input_data[value] = 1 if input_data['st_slope'].iloc[0] == key else 0
    
    input_data['Sex_M'] = sex_mapping[input_data['sex'].iloc[0]]
    

    input_data.drop(['sex', 'chest_pain_type', 'ecg', 'exercise_angina', 'st_slope'], axis=1, inplace=True)
    
    input_data.rename(columns={
        'age': 'Age', 'max_hr': 'MaxHR', 'cholesterol': 'Cholesterol',
        'resting_bp': 'RestingBP', 'old_peak': 'Oldpeak', 'fasting_bs': 'FastingBS'
    }, inplace=True)
    
    expected_columns = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
 
    input_data = input_data[expected_columns]
    
    return input_data

@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        
        input_data = {
            'age': float(request.form['age']),
            'max_hr': float(request.form['maxhr']),
            'cholesterol': float(request.form['cholesterol']),
            'resting_bp': float(request.form['restingbp']),
            'old_peak': float(request.form['oldpeak']),
            'fasting_bs': float(request.form['fastingbs']),
            'sex': request.form['sex'],
            'ecg': request.form['ecg'],
            'exercise_angina': request.form['exerciseangina'],
            'st_slope': request.form['stslope'],
            'chest_pain_type': request.form['chestpaintype']
        }

      
        input_df = pd.DataFrame([input_data])
        input_df = transform_input_data(input_df)

        trained_model = train_xgboost_model()

       
        prediction_value = trained_model.predict(input_df)

        
        result = 'There is a risk of heart disease.' if prediction_value == 1 else 'There is no risk of heart disease.'

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
