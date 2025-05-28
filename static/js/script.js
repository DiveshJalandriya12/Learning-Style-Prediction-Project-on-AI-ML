document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded and parsed');
    
    var _d = document;

    _d.getElementById('prediction-form').addEventListener('submit', function(e) {
        e.preventDefault();  // Prevent the default form submission

        console.log('Form submission intercepted');

        // Collect form data and convert to JSON
        const formData = new FormData(this);
        const formJSON = Object.fromEntries(formData.entries());

        console.log('Form Data:', formJSON);

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formJSON)
        })
        .then(response => response.json())
        .then(data => {
            console.log('Server Response:', data);
            if (data.error) {
                _d.getElementById('result').innerText = "Error: " + data.error;
            } else {
                let resultHTML = '<h3>Predictions:</h3>';
                resultHTML += <p>Predicted Learning Style: ${data.predicted_style}</p>;

                resultHTML += <><p>Probabilities:</p><ul>;
                    for (const [style, value] of Object.entries(data.probabilities)) {resultHTML += <li>${style}: ${(value * 100).toFixed(2)}%</li>};
                    {"}"}
                    resultHTML += </ul></>;

                _d.getElementById('result').innerHTML = resultHTML;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            _d.getElementById('result').innerText = "Error: " + error.message;
        });
    });
});



