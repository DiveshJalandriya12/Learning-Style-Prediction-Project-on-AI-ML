from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

# Load the question dataset
questions_df = pd.read_csv('questions.csv')

# Load the trained model
model = joblib.load('learning_style_model.pkl')

# Define the question categories and their corresponding indices
categories = {
    'auditory': range(0, 10),
    'kinesthetic': range(10, 20),
    'visual': range(20, 30)
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user's responses
        responses = [int(request.form[f'Q{i+1}']) for i in range(30)]

        # Make predictions using the trained model
        predicted_style = model.predict([responses])[0]

        # Get the predicted learning style
        le = LabelEncoder()
        le.classes_ = np.array(['auditory', 'kinesthetic', 'visual'])
        predicted_style = le.inverse_transform([predicted_style])[0]

        # Store the predicted learning style in the session
        session['predicted_style'] = predicted_style

        # Redirect to the results page
        return render_template('results.html', predicted_style=predicted_style)

    # If the request method is GET, render the index page with random questions
    auditory_questions = questions_df[questions_df['Category'].str.lower() == 'auditory']
    visual_questions = questions_df[questions_df['Category'].str.lower() == 'kinesthetic']
    kinesthetic_questions = questions_df[questions_df['Category'].str.lower() == 'visual']

    random_questions = pd.concat([
        auditory_questions.sample(min(10, len(auditory_questions))),
        visual_questions.sample(min(10, len(visual_questions))),
        kinesthetic_questions.sample(min(10, len(kinesthetic_questions)))
    ]).reset_index(drop=True)

    return render_template('index.html', questions=random_questions)

@app.route('/results')
def results():
    predicted_style = session.get('predicted_style')
    return render_template('results.html', predicted_style=predicted_style)

if __name__ == '__main__':
    app.run(debug=True)