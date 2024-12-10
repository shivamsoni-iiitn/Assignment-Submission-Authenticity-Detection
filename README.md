This repository contains a Flask application (app.py) that you can run on your local machine. Follow these steps to set up the environment and dependencies:

1. Install Python 3.10
First, ensure that you have Python 3.10 installed on your local machine. If you don't have it yet, you can download and install it from the official Python website.
Note: I've already uploaded the Python 3.10 version in this repository, so you don't need to worry if you don't have it yet.

2. Create a Virtual Environment
Once Python 3.10 is installed, you need to create a new virtual environment. This helps keep the project dependencies isolated from your system's Python installation.
Run the following command in your terminal or command prompt:
**python3.10 -m venv my_new_env**
This will create a new virtual environment named my_new_env.

3. Activate the Virtual Environment
Next, activate the virtual environment.

For Windows:
**my_new_env\Scripts\activate**
For macOS/Linux:
**source my_new_env/bin/activate** or **chmod +x my_new_env/bin/activate**

4. Install Project Dependencies
Now that your virtual environment is activated, install the required Python packages:
**pip install flask numpy transformers scikit-learn joblib torch PyPDF2 docx2txt**
This will install all the necessary libraries to run the Flask app.

6. Run the Application
Once the dependencies are installed, you can run the application using the following command:
**python app.py**
The Flask app should now be running locally on your machine.
