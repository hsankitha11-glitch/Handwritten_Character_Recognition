# Handwritten_Character_Recognition

# Handwritten Character Recognition using CNN

## Project Overview
This project is a Handwritten Character Recognition System developed using Python, TensorFlow, and Convolutional Neural Networks (CNN). The model is trained on the MNIST dataset to recognize handwritten digits (0–9). It predicts a random handwritten digit each time the program is executed.

## Features
- Recognizes handwritten digits from 0 to 9.
- Uses the MNIST dataset for training and testing.
- Implements a Convolutional Neural Network (CNN).
- Achieves high prediction accuracy.
- Displays the predicted digit along with the corresponding image.
- Saves the trained model for future use.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Dataset
- MNIST Handwritten Digits Dataset
- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Classes: Digits (0–9)

## CNN Architecture
- Convolution Layer (32 filters)
- Max Pooling Layer
- Convolution Layer (64 filters)
- Max Pooling Layer
- Flatten Layer
- Dense Layer (128 neurons)
- Output Layer (10 neurons with Softmax activation)

## Project Workflow
1. Load the MNIST dataset.
2. Preprocess the images by normalizing pixel values.
3. Convert labels into one-hot encoded format.
4. Build the CNN model.
5. Train the model.
6. Evaluate the model on the test dataset.
7. Randomly select one handwritten digit.
8. Predict and display the result.
9. Save the trained model.

## Installation

Install the required libraries:

```bash
pip install tensorflow matplotlib numpy
```

## Run the Project

```bash
python handwritten_character_recognition.py
```

## Expected Output
- Displays the test accuracy.
- Predicts one random handwritten digit.
- Shows the handwritten image.
- Saves the trained model as `handwritten_digit_model.h5`.

## Project Structure

```
HorizonTechX_HandwrittenCharacterRecognition/
│── handwritten_character_recognition.py
│── handwritten_digit_model.h5
│── README.md
```

## Applications
- Optical Character Recognition (OCR)
- Bank cheque digit recognition
- Postal code recognition
- Document digitization
- Handwritten form processing

## Future Enhancements
- Recognize handwritten alphabets using the EMNIST dataset.
- Recognize complete words and sentences.
- Develop a GUI using Tkinter.
- Create a web application using Flask or Streamlit.
- Predict handwritten digits from user-uploaded images.

## Author
Ankitha H S

## Internship
Machine Learning Internship – Horizon TechX
