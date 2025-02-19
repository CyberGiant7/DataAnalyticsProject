# Data Analytics Project

## Description

This project is a comprehensive data analytics project that involves data preprocessing, model training, and evaluation. The project is structured into different modules, including a TrainingModule and a TestModule, each with specific functionalities.

### Project Structure

#### TrainingModule

- `ff_module.ipynb`: This notebook is used for training a Feedforward Neural Network. It includes data preprocessing, model definition, training, and evaluation.
- `tabnet_module.ipynb`: This notebook is used for training a TabNet model. It includes data preprocessing, model definition, training, and evaluation.
- `tabtransformer_module.ipynb`: This notebook is used for training a TabTransformer model. It includes data preprocessing, model definition, training, and evaluation.
- `data_visualization_module.ipynb`: This notebook is used for visualizing the data. It includes various plots and charts to understand the data better.
- `train_module.ipynb`: This notebook is used for training various models. It includes data preprocessing, model definition, training, and evaluation.

#### TestModule

- `test.py`: This script is used for testing the trained models. It includes functions for loading the models, preprocessing the test data, and evaluating the model performance.
- `requirements.txt`: This file contains the list of dependencies required for the TestModule.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r TestModule/requirements.txt
```

## Usage

### Training

To train the models, navigate to the `TrainingModule` directory and run the respective Jupyter notebooks:

- `ff_module.ipynb`: Feedforward Neural Network training
- `tabnet_module.ipynb`: TabNet model training
- `tabtransformer_module.ipynb`: TabTransformer model training
- `data_visualization_module.ipynb`: Data visualization
- `train_module.ipynb`: Training various models

### Testing

To test the models, navigate to the `TestModule` directory and run the `test.py` script:

```bash
python test.py
```

## Contributing

We welcome contributions to this project. Please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
