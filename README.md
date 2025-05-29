```markdown
# Yoga Pose Detection with Web UI

## Overview

This project develops a deep learning model to classify images into different yoga poses. It includes a Jupyter Notebook for training the model and a Flask web application to provide a user interface for uploading images and getting pose predictions. The system can identify poses such as Downward Dog, Goddess, Plank, Tree, and Warrior.

## Features

*   **Yoga Pose Classification Model:** Trains a Convolutional Neural Network (CNN) using transfer learning with the Xception model to classify yoga poses.
*   **Jupyter Notebook for Training:** Provides a `Yoga-Pose-Training-Local.ipynb` notebook for step-by-step model training and experimentation on a local machine.
*   **Web Interface:** A simple web UI built with Flask and HTML/JavaScript allows users to upload an image and receive a prediction of the yoga pose.
*   **Confidence Score:** Displays the confidence level of the prediction.
*   **Handles Irrelevant Images:** If the model's confidence for a predicted pose is below a certain threshold (60%), it suggests the image might not be a recognized yoga pose or is unclear.

## Directory Structure

```
.
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
├── Yoga-Pose-Training-Local.ipynb # Jupyter Notebook for model training
├── yoga_pose_model.h5    # (Generated after training) The trained Keras model
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup Instructions

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

Install the required Python packages using pip:

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

## Running the Application

### 1. Train the Model (First Time)

Before running the web application, you need to train the model:

1.  Start a Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
2.  Open `Yoga-Pose-Training-Local.ipynb` in your browser.
3.  Run all the cells in the notebook. This will train the model and save it as `yoga_pose_model.h5` in the project's root directory. This process might take some time, especially if you are not using a GPU.

### 2. Run the Flask Web Application

Once the `yoga_pose_model.h5` file is generated:

1.  Open your terminal, navigate to the project's root directory (if you're not already there), and ensure your virtual environment is activated.
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to: `http://127.0.0.1:5000/`

## How to Use

1.  Access the web application via the URL provided when you run `app.py`.
2.  Click the "Choose File" button to select an image of a yoga pose.
3.  A preview of the selected image will appear.
4.  Click the "Predict Pose" button.
5.  The application will display the predicted yoga pose and the confidence score. If the confidence is below 60%, it will indicate that the image might not be a recognized pose or is unclear.

## Troubleshooting

*   **`FileNotFoundError: [Errno 2] No such file or directory: 'yoga_pose_model.h5'`**: This means the model file was not found. Make sure you have successfully run the `Yoga-Pose-Training-Local.ipynb` notebook to train and save the model.
*   **TensorFlow GPU Issues / `Could not load dynamic library 'cudart64_...dll'` or similar**:
    *   If you don't have a compatible NVIDIA GPU or CUDA/cuDNN installed, TensorFlow might try to use the GPU and fail.
    *   You can try installing the CPU-only version of TensorFlow: `pip install tensorflow-cpu`. Remember to uninstall the GPU version first (`pip uninstall tensorflow tensorflow-gpu`) if it was installed.
    *   Alternatively, ensure your NVIDIA drivers, CUDA Toolkit, and cuDNN library are correctly installed and compatible with your TensorFlow version.
*   **Dataset Path Issues (`Found 0 images belonging to 0 classes.`)**:
    *   Double-check that your `INPUT_DATASET` directory is in the same directory as `Yoga-Pose-Training-Local.ipynb` and `app.py`.
    *   Ensure the `TRAIN` and `TEST` folders are directly inside `INPUT_DATASET` and contain the respective class subfolders (`downdog`, `goddess`, etc.) with images.
*   **Pillow/PIL Errors (e.g., "cannot identify image file")**:
    *   Some images in the dataset might be corrupted. The `removeCorruptedImages` function in the notebook can help, but ensure it's run correctly on the dataset directories.
    *   Make sure you have the latest version of Pillow: `pip install --upgrade Pillow`.

```
