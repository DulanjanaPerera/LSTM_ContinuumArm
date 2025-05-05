# LSTM\_ContinuumArm

This repository contains code for modeling and controlling a continuum robotic arm using Long Short-Term Memory (LSTM) networks. The project focuses on mapping between different parameter spaces—such as configuration, pressure, and task space—using forward and inverse models, complemented by Bayesian optimization techniques.

## 📁 Project Structure

### `ConfigToTask`

This module handles the mapping between configuration parameters and task space.

* \`\`: Implements the forward model that maps configuration parameters to task space.
* \`\`: Performs inverse modeling and applies Bayesian optimization to map task space goals back to configuration parameters.
* \`\`: Loads a pre-trained model to validate and test inverse mapping performance.
* **Other **\`\`** files**: Include scripts for dataset generation (both forward and inverse) and the LSTM model class definitions.

### `PressureToTask`

This module focuses on the relationship between pressure inputs and task space.

* \`\`: Develops a forward model using Bayesian optimization to map pressure inputs to task space.
* \`\`: Implements inverse modeling with Bayesian optimization to determine pressure inputs required for desired task space outcomes.
* \`\`: Fine-tunes the inverse model using additional data to improve accuracy.
* \`\`: Tests the system's ability to follow predefined trajectories in task space.

## 🛠️ Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DulanjanaPerera/LSTM_ContinuumArm.git
   cd LSTM_ContinuumArm
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   Ensure you have Python 3.6 or higher. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If **`requirements.txt`** is not provided, manually install the necessary packages such as **`numpy`**, **`torch`**, **`scikit-learn`**, and **`matplotlib`**.*

## 🚀 Usage

### Configuration to Task Mapping

* **Forward Model**

  To run the forward model that maps configuration parameters to task space:

  ```bash
  cd ConfigToTask
  python main.py
  ```

* **Inverse Modeling with Bayesian Optimization**

  To perform inverse modeling and apply Bayesian optimization:

  ```bash
  python main_inv_opt.py
  ```

* **Reload and Test Trained Model**

  To load a pre-trained model and test its performance:

  ```bash
  python main_inv_opt_reload.py
  ```

### Pressure to Task Mapping

* **Forward Model with Bayesian Optimization**

  To develop a forward model mapping pressure inputs to task space:

  ```bash
  cd PressureToTask
  python main_opti.py
  ```

* **Inverse Modeling**

  To perform inverse modeling to determine required pressure inputs:

  ```bash
  python main_opt_inv.py
  ```

* **Fine-Tune Inverse Model**

  To fine-tune the inverse model with additional data:

  ```bash
  python main_opt_inv_finetune.py
  ```

* **Trajectory Tracking**

  To test the system's ability to follow predefined trajectories:

  ```bash
  python main_trajectory_tracking.py
  ```

## 📄 Documentation

For detailed explanations of the methodologies, experiments, and results, please refer to the accompanying PDF document:

* [Project Documentation](./docs/LSTM_ContinuumArm_Report.pdf)

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or encounter issues, please open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*For any questions or further information, please contact *[*Dulanjana Perera*](mailto:your_email@example.com)*.*
