# Student Grades Prediction with Keras

This project implements a Deep Learning regression model using Keras to predict student scores based on their study habits and exam performances.

## Dataset

The model uses the `Student_Grades.csv` dataset.

* **Features (Inputs):** Hours, Practice, TeamWork, MidTerm, FinalExam
* **Target (Output):** Scores
* **Excluded:** Grade (Categorical variable removed during preprocessing)

## Dependencies

* Python 3.x
* Pandas
* NumPy
* Keras / TensorFlow

## Model Architecture

The model is built using the Keras `Sequential` API with the following structure:

1. **Dense Layer:** 16 neurons
2. **Dense Layer:** 32 neurons, Activation: `relu`
3. **Dense Layer:** 32 neurons, Activation: `relu`
4. **Output Layer:** 1 neuron (Regression output)

## Training Configuration

* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE)
* **Epochs:** 128
* **Batch Size:** 16

## Usage

### Data Preprocessing
The data is loaded using Pandas. The `Scores` column is separated as the target variable (y), and the `Grade` column is dropped. All values are cast to `float32`.

### Training
```python
model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=128, batch_size=16)
```

### Prediction
After training, the model can predict scores for new inputs. Example input: [Hours: 5, Practice: 1, TeamWork: 0.5, MidTerm: 4, FinalExam: 4.2]
```prediction = model.predict(np.array([[5, 1, 0.5, 4, 4.2]]))```
# Output: Predicted Score
