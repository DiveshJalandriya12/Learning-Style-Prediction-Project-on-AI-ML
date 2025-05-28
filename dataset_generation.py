import pandas as pd
import numpy as np

# Define the number of users (rows) and questions (columns)
n_users = 20000
n_questions = 30

# Define ranges for each learning style in the questions
auditory_indices = range(0, 10)
visual_indices = range(10, 20)
kinesthetic_indices = range(20, 30)

# Function to generate a random response biased towards a particular style
def generate_responses(bias_style):
    responses = np.random.randint(0, 4, size=n_questions)
    
    if bias_style == 'Auditory':
        responses[auditory_indices] = np.random.randint(2, 4, size=10)  # Higher chance of Agree/Slightly Agree
    elif bias_style == 'Visual':
        responses[visual_indices] = np.random.randint(2, 4, size=10)
    elif bias_style == 'Kinesthetic':
        responses[kinesthetic_indices] = np.random.randint(2, 4, size=10)
    
    return responses

# Generate data for 20000 users
data = []
for _ in range(n_users):
    # Randomly choose a dominant learning style
    dominant_style = np.random.choice(['Auditory', 'Visual', 'Kinesthetic'])
    
    # Generate responses with a bias towards the dominant style
    responses = generate_responses(dominant_style)
    
    # Calculate total scores for each learning style
    auditory_score = sum(responses[auditory_indices])
    visual_score = sum(responses[visual_indices])
    kinesthetic_score = sum(responses[kinesthetic_indices])
    
    # Determine the predicted learning style based on scores
    scores = {'Auditory': auditory_score, 'Visual': visual_score, 'Kinesthetic': kinesthetic_score}
    predicted_style = max(scores, key=scores.get)
    
    # Append the responses and predicted learning style to the data
    data.append(np.append(responses, predicted_style))

# Convert the data to a DataFrame
columns = [f'Q{i+1}' for i in range(n_questions)] + ['Learning Style']
df = pd.DataFrame(data, columns=columns)

# Save the dataset to a CSV file
df.to_csv('learning_style_dataset.csv', index=False)

print("Sample data:\n", df.head())
