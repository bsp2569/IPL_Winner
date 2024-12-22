from flask import Flask, render_template, request
import pandas as pd
from joblib import load  # Using joblib for better model serialization

app = Flask(__name__)

# Load the trained model
pipe = load('ra_pipe.joblib')  # Ensure the correct model file is loaded

@app.route('/')
def home():
    return render_template('index.html')  # Renders the frontend form

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input values from the form
            batting_team = request.form['batting_team']
            bowling_team = request.form['bowling_team']
            selected_city = request.form['selected_city']
            target = int(request.form['target'])
            score = int(request.form['score'])
            balls_left = int(request.form['balls_left'])
            wickets = int(request.form['wickets'])

            # Calculate additional features
            runs_left = target - score
            wickets_remaining = 10 - wickets
            overs_completed = (120 - balls_left) / 6
            crr = score / overs_completed
            rrr = runs_left / (balls_left / 6)

            # Prepare the input data for the model
            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [selected_city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_remaining': [wickets_remaining],
                'total_run_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            # Predict probabilities using the model
            result = pipe.predict_proba(input_data)
            win_probability = round(result[0][1] * 100)
            loss_probability = round(result[0][0] * 100)

            # Render the result template with predictions
            return render_template(
                'result.html',
                batting_team=batting_team,
                bowling_team=bowling_team,
                win_probability=win_probability,
                loss_probability=loss_probability
            )
        except Exception as e:
            # Handle any errors gracefully
            return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
