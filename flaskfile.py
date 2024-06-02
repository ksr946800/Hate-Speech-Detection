from flask import Flask, request, jsonify
import another_copy_of_untitled8

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  text = data['text']

  # Call your model prediction function and get the class
  predicted_class = your_hate_speech_detection_model.predict_class(text)

  return jsonify({'class': predicted_class})

if __name__ == '__main__':
  app.run(debug=True)