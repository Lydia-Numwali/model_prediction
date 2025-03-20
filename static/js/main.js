// Function to create form fields based on feature names
function createFormFields(featureNames) {
    const form = document.getElementById('predictionForm');
    form.innerHTML = '';
    
    featureNames.forEach(feature => {
        const div = document.createElement('div');
        div.className = 'form-group mb-3';
        
        const label = document.createElement('label');
        label.textContent = feature;
        label.className = 'form-label';
        
        const select = document.createElement('select');
        select.className = 'form-select';
        select.name = feature;
        select.required = true;
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select a value';
        select.appendChild(defaultOption);
        
        // Add options based on feature type
        if (feature === 'Age') {
            // Add age options from 18 to 30
            for (let i = 18; i <= 30; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = i;
                select.appendChild(option);
            }
        } else if (feature === 'CGPA') {
            // Add CGPA options from 2.0 to 4.0
            for (let i = 20; i <= 40; i++) {
                const option = document.createElement('option');
                const value = (i / 10).toFixed(1);
                option.value = value;
                option.textContent = value;
                select.appendChild(option);
            }
        } else if (feature === 'Depression' || feature === 'Family History') {
            // Add Yes/No options
            ['Yes', 'No'].forEach(value => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                select.appendChild(option);
            });
        } else {
            // Add options from feature_values
            const values = featureValues[feature] || [];
            values.forEach(value => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                select.appendChild(option);
            });
        }
        
        div.appendChild(label);
        div.appendChild(select);
        form.appendChild(div);
    });
}

// Function to get CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Function to make prediction
async function makePrediction() {
    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const data = {};
    
    // Get all form fields and their values
    for (let [key, value] of formData.entries()) {
        if (!value) {
            alert('Please fill in all fields');
            return;
        }
        
        // Convert string values to appropriate types
        if (key === 'Age') {
            data[key] = parseInt(value);
        } else if (key === 'CGPA') {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }
    
    try {
        console.log('Sending data:', data); // Debug log
        
        const response = await fetch('/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }
        
        const result = await response.json();
        console.log('Received result:', result); // Debug log
        
        displayPredictions(result);
        createVisualization(result);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('predictions').innerHTML = `
            <div class="alert alert-danger">
                Error: ${error.message}
            </div>
        `;
    }
}

// Function to get level description
function getLevelDescription(level) {
    if (level <= 1) return 'Low';
    if (level <= 2) return 'Moderate';
    if (level <= 3) return 'High';
    return 'Very High';
}

// Function to get level color
function getLevelColor(level) {
    if (level <= 1) return '#4CAF50';  // Green for Low
    if (level <= 2) return '#FFC107';  // Yellow for Moderate
    if (level <= 3) return '#FF9800';  // Orange for High
    return '#F44336';  // Red for Very High
}

// Function to display predictions
function displayPredictions(result) {
    const predictionsDiv = document.getElementById('predictions');
    let html = '<h5>Mental Health Assessment:</h5>';
    
    result.predictions.forEach((prediction, index) => {
        const feature = result.feature_names[index];
        const level = Math.round(prediction);
        const description = getLevelDescription(prediction);
        const color = getLevelColor(prediction);
        
        html += `
            <div class="prediction-item mb-3">
                <h6>${feature}</h6>
                <div class="d-flex align-items-center">
                    <div class="level-indicator me-3" style="background-color: ${color};"></div>
                    <div>
                        <strong>${description}</strong>
                        <span class="text-muted ms-2">(${prediction.toFixed(2)})</span>
                    </div>
                </div>
            </div>
        `;
    });
    
    predictionsDiv.innerHTML = html;
}

// Function to create visualization
function createVisualization(result) {
    const trace = {
        x: result.feature_names,
        y: result.predictions,
        type: 'bar',
        text: result.predictions.map(pred => getLevelDescription(pred)),
        textposition: 'auto',
        marker: {
            color: result.predictions.map(pred => getLevelColor(pred))
        }
    };

    const layout = {
        title: 'Mental Health Assessment Results',
        xaxis: {
            title: 'Mental Health Indicators',
            tickangle: 45
        },
        yaxis: {
            title: 'Level',
            range: [0, 4],
            ticktext: ['Low', 'Moderate', 'High', 'Very High'],
            tickvals: [0, 1, 2, 3, 4]
        },
        height: 400,
        margin: {
            b: 100
        },
        showlegend: false
    };

    Plotly.newPlot('visualization', [trace], layout);
}

// Initialize form fields when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Get feature names from the model
    const featureNames = ['Age', 'CGPA', 'City', 'Degree', 'Depression', 'Dietary Habits', 
                         'Exercise', 'Family History', 'Gender', 'Income', 'Marital Status',
                         'Physical Health', 'Sleep Quality', 'Social Support', 'Substance Use'];
    
    // Define feature values
    const featureValues = {
        'Age': Array.from({length: 13}, (_, i) => i + 18),  // 18 to 30
        'CGPA': Array.from({length: 21}, (_, i) => ((i + 20) / 10).toFixed(1)),  // 2.0 to 4.0
        'City': ['Urban', 'Suburban', 'Rural'],
        'Degree': ['Bachelor', 'Master', 'PhD'],
        'Depression': ['Yes', 'No'],
        'Dietary Habits': ['Poor', 'Fair', 'Good'],
        'Exercise': ['Never', 'Sometimes', 'Regular'],
        'Family History': ['Yes', 'No'],
        'Gender': ['Male', 'Female', 'Other'],
        'Income': ['Low', 'Medium', 'High'],
        'Marital Status': ['Single', 'Married', 'Divorced'],
        'Physical Health': ['Poor', 'Fair', 'Good'],
        'Sleep Quality': ['Poor', 'Fair', 'Good'],
        'Social Support': ['Low', 'Medium', 'High'],
        'Substance Use': ['Never', 'Sometimes', 'Regular']
    };
    
    window.featureValues = featureValues;  // Make featureValues available globally
    createFormFields(featureNames);
}); 