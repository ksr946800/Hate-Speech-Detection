const userInputElement = document.getElementById('user-input');
const predictButton = document.getElementById('predict-button');
const predictionElement = document.getElementById('prediction');

predictButton.addEventListener('click', async () => {
  const userInput = userInputElement.value;

  // Replace with actual fetch request to your Python server
  
  const response = await fetch('/predict', {
    method: 'POST',
    body: JSON.stringify({ text: userInput }),
    headers: { 'Content-Type': 'application/json' }
  });

  const predictionData = await response.json();
  predictionElement.textContent = `Prediction: ${predictionData.class}`;
});