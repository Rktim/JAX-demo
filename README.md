
## Linear Regression with JAX

### Objective
This project demonstrates the application of the JAX library for performing linear regression on a synthetic dataset.

### Methods
1. **Data Generation:**
    - Generate 100 random samples of `x` uniformly distributed between 0 and 1.
    - Define the true model parameters: `trueW` and `trueB`.
    - Generate the target values `y` by adding Gaussian noise to the true model output.

2. **Model Definition:**
    - Define a `predict` function that takes the model parameters `w` and `b` and an input `x`, and returns the predicted value.

3. **Loss Function:**
    - Define a `loss` function that takes the model parameters, input data `x`, and target values `y`, and returns the mean squared error between the predicted and true values.

4. **Gradient Calculation:**
    - Use JAX's `grad` function to automatically compute the gradients of the loss function with respect to the model parameters.

5. **Parameter Update:**
    - Initialize the model parameters `w` and `b` with random values.
    - Perform 1000 iterations of gradient descent using a learning rate of 0.1.
    - Update the parameters in each iteration using the calculated gradients.

6. **Evaluation:**
    - Print the final loss, trained weights, and trained bias.
    - Plot the data points and the fitted line.

### Expected Outcomes
- The trained weights and bias should be close to the true values used for data generation.
- The fitted line should closely resemble the true model.

### Benefits
This project showcases the use of JAX for defining, training, and evaluating a linear regression model. JAX offers several advantages, including:
- Automatic differentiation for efficient gradient calculations.
- JIT compilation for improved performance.
- Support for various hardware accelerators, such as GPUs and TPUs.

### Future Work
This project can be extended in several ways:
- Experiment with different learning rates and the number of iterations.
- Implement other regression models, such as polynomial regression or logistic regression.
- Apply JAX to more complex machine learning tasks, such as image classification or natural language processing.

### Technologies Used
- **Python**: The primary programming language used for this project.
- **JAX**: A library for high-performance numerical computing and machine learning research.
- **NumPy**: A library for numerical operations in Python.
- **Matplotlib**: A plotting library for visualizing data and results.



**View the Results:**
    - The final loss, trained weights, and trained bias will be printed in the console.
    - A plot of the data points and the fitted line will be displayed.

### Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.


### Acknowledgements
- Special thanks to the developers of JAX, NumPy, and Matplotlib for providing the essential tools used in this project.

Happy coding!
