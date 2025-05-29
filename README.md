# Deep Learning based Yoga Pose Detection with Web UI
<p align="center">
  <img src="https://github.com/akhilchibber/Yoga-Pose-Detection/blob/main/YOGA_POSE_DETECTION.jpg?raw=true" alt="Yoga Pose Detection Banner">
</p>

## Overview
This repository contains a project focused on developing a deep learning model to classify images into different yoga poses. It includes a Jupyter Notebook (`Yoga-Pose-Training-Local.ipynb`) for training the model on your local machine and a Flask web application that provides a user-friendly interface for uploading images and getting real-time pose predictions. The project uses convolutional neural networks (CNNs) and transfer learning techniques with the Xception model to distinguish among five yoga poses: downward dog, goddess, plank, tree, and warrior.

## Features

### Core Model & Training:
*   **Yoga Pose Classification Model:** Trains a CNN using transfer learning from the Xception model.
*   **Jupyter Notebook for Local Training:** `Yoga-Pose-Training-Local.ipynb` provides a step-by-step guide for model training and experimentation.
*   **Dataset:** Utilizes the "Yoga Poses Dataset" from Kaggle.

### Web Application:
*   **User-Friendly Interface:** A simple web UI built with Flask and HTML/JavaScript for easy image uploading and prediction.
*   **Real-time Prediction:** Get instant classification of the uploaded yoga pose.
*   **Confidence Score:** Displays the confidence level of the prediction.
*   **Handles Irrelevant Images:** If the model's confidence for a predicted pose is below 60%, it suggests the image might not be a recognized yoga pose or is unclear.

## Directory Structure
├── INPUT_DATASET/        # (User-created) Directory for training and test images
│   ├── TRAIN/
│   │   ├── downdog/
│   │   ├── goddess/
│   │   ├── plank/
│   │   ├── tree/
│   │   └── warrior2/
│   └── TEST/
│       ├── downdog/
│       ├── goddess/
│       ├── plank/
│       ├── tree/
│       └── warrior2/
├── templates/
│   └── index.html        # Frontend HTML for the web application
├── app.py                # Flask backend application
├── Yoga-Pose-Detection.ipynb # Original Colab-based notebook (for reference)
├── Yoga-Pose-Training-Local.ipynb # Jupyter Notebook for local model training
├── yoga_pose_model.h5    # (Generated after training) The trained Keras model
├── requirements.txt      # Python dependencies
└── README.md             # This file

## Setup and Running the Application

### 1. Prerequisites

*   **Python 3.x:** Ensure you have Python 3 installed (Python 3.7+ recommended).
*   **pip:** Python package installer.
*   **Kaggle API Key:** You'll need a Kaggle account and an API key (`kaggle.json`) to download the dataset.
    *   Follow the instructions here to get your API key: [Kaggle API Documentation](https://www.kaggle.com/docs/api)
    *   Place the `kaggle.json` file in the appropriate directory:
        *   Linux/macOS: `~/.kaggle/kaggle.json`
        *   Windows: `C:\Users\<Your-Username>\.kaggle\kaggle.json`
    *   Make sure the file has appropriate permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/macOS).

### 2. Clone the Repository (or Download Files)

If this project is in a Git repository:
```bash
git clone <repository-url>
cd <repository-name>
```
Otherwise, download and extract the project files into a local directory.

### 3. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS and Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Download the Dataset

The dataset is downloaded from Kaggle using the Kaggle API.

1.  Ensure your Kaggle API key is set up (see Prerequisites).
2.  Run the following command in your terminal from the project's root directory. This will download and unzip the dataset into the `INPUT_DATASET` folder.
    ```bash
    mkdir INPUT_DATASET
    kaggle datasets download -d niharika41298/yoga-poses-dataset -p INPUT_DATASET --unzip
    ```
3.  After unzipping, the dataset might be inside a subfolder (e.g., `yoga_dataset_v1` or `Dataset`). You need to ensure the `TRAIN` and `TEST` directories are directly under `INPUT_DATASET` as shown in the "Directory Structure" section. You might need to move these folders.
    For example, if after unzipping you have `INPUT_DATASET/yoga_dataset_v1/TRAIN` and `INPUT_DATASET/yoga_dataset_v1/TEST`, move them so you have `INPUT_DATASET/TRAIN` and `INPUT_DATASET/TEST`.

    **Note on Corrupted Images:** The original notebook includes a function to remove corrupted images. If you encounter issues during training (e.g., "cannot identify image file"), you might need to run the `removeCorruptedImages` function from the `Yoga-Pose-Training-Local.ipynb` notebook after setting up the dataset. Ensure the paths in the notebook are correct if you do so.

### 6. Train the Model (First Time)

Before running the web application, you need to train the model:

1.  Start a Jupyter Notebook server in your activated virtual environment:
    ```bash
    jupyter notebook
    ```
2.  Open `Yoga-Pose-Training-Local.ipynb` in your browser.
3.  Run all the cells in the notebook. This will train the model and save it as `yoga_pose_model.h5` in the project's root directory. This process might take some time, especially if you are not using a GPU.

### 7. Run the Flask Web Application

Once the `yoga_pose_model.h5` file is generated:

1.  Open your terminal, navigate to the project's root directory (if you're not already there), and ensure your virtual environment is activated.
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to: `http://127.0.0.1:5000/`

## How to Use the Web Application

1.  Access the web application via the URL (`http://127.0.0.1:5000/`) provided when you run `app.py`.
2.  Click the "Choose File" button to select an image of a yoga pose from your local system.
3.  A preview of the selected image will appear on the page.
4.  Click the "Predict Pose" button.
5.  The application will display the predicted yoga pose and the confidence score. If the confidence is below 60%, it will indicate that the image might not be a recognized pose or is unclear, while still showing the most likely prediction.

## Troubleshooting

*   **`FileNotFoundError: [Errno 2] No such file or directory: 'yoga_pose_model.h5'`**: This means the model file was not found by `app.py`. Make sure you have successfully run all cells in the `Yoga-Pose-Training-Local.ipynb` notebook, which includes the model training and saving steps. The `yoga_pose_model.h5` file should be in the same directory as `app.py`.
*   **TensorFlow GPU Issues / `Could not load dynamic library 'cudart64_...dll'` or similar**:
    *   If you don't have a compatible NVIDIA GPU or CUDA/cuDNN installed, TensorFlow might try to use the GPU and fail.
    *   You can try installing the CPU-only version of TensorFlow: `pip install tensorflow-cpu`. Remember to uninstall the GPU version first (`pip uninstall tensorflow tensorflow-gpu`) if it was installed.
    *   Alternatively, ensure your NVIDIA drivers, CUDA Toolkit, and cuDNN library are correctly installed and compatible with your TensorFlow version.
*   **Dataset Path Issues (`Found 0 images belonging to 0 classes.`) in the Notebook**:
    *   Double-check that your `INPUT_DATASET` directory is in the same directory as `Yoga-Pose-Training-Local.ipynb`.
    *   Ensure the `TRAIN` and `TEST` folders are directly inside `INPUT_DATASET` and contain the respective class subfolders (`downdog`, `goddess`, etc.) with images. The structure should be exactly as specified in the "Directory Structure" and setup instructions.
*   **Pillow/PIL Errors (e.g., "cannot identify image file") during training or prediction**:
    *   Some images in the downloaded dataset might be corrupted. The `Yoga-Pose-Training-Local.ipynb` notebook contains a (commented out by default) `removeCorruptedImages` function. You can uncomment and run this cell to attempt to clean the dataset.
    *   Ensure you have the latest version of Pillow: `pip install --upgrade Pillow`.
*   **Kaggle API Issues**:
    *   Ensure `kaggle.json` is correctly placed and has the right permissions.
    *   Make sure you have accepted the rules for the "Yoga Poses Dataset" on the Kaggle website.

## Target Audience (from original notebook)
This Github Repository, specifically the Jupyter notebook, is structured to be accessible for beginners, providing detailed explanations and a step-by-step approach, while also encompassing advanced techniques for seasoned practitioners. Let's dive into the world of Yoga Pose Detection and deep learning!

## Contributing

We welcome contributions to enhance the functionality and efficiency of this script. Feel free to fork, modify, and make pull requests to this repository. To contribute:

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request against the `main` branch.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

Author: Akhil Chhibber

LinkedIn: https://www.linkedin.com/in/akhilchhibber/

Medium Blogs: https://medium.com/@akhil.chibber
```
