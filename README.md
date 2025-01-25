# Product-Quality-Detection-System

# AI-Based Fruit Classification System with Real-Time Quality Assessment 

**Overview**
AI-Based Fruit Classification System with Real-Time Quality Assessment
Overview
This project implements an AI-based Fruit Classification System that can identify fruit types and assess their quality (fresh or rotten) based on image analysis. The system utilizes deep learning models, specifically Convolutional Neural Networks (CNNs), for accurate fruit classification and quality detection. This application is designed to be useful for industries such as agriculture, retail, and e-commerce, where the quality of products plays a crucial role in business operations.

The system is built using Flask for the web interface and TensorFlow/Keras for the machine learning model. It supports both image uploads and real-time image analysis through camera input, allowing users to classify fruits directly from their devices.
![Image](https://github.com/user-attachments/assets/1eefca2b-3396-4582-a574-9bd6f07b4cc5)

**Features**

**Fruit Type Classification:** Classifies fruits into various categories like Apple, Banana, Orange, Mango, etc.
Quality Detection: Assesses the quality of fruits, categorizing them as Fresh, Medium, or Rotten.
**Real-time Feedback:** Allows users to provide feedback on the classification results, improving the model's accuracy over time.
Web Interface: Built with Flask, providing an easy-to-use interface for users to upload images or use their cameras for analysis.
**Dashboard:** Displays real-time statistics about the classification results and user feedback.
![Image](https://github.com/user-attachments/assets/22f640b5-d461-4191-9fc3-13d9bda7c661)
![Image](https://github.com/user-attachments/assets/79de69e0-f15e-439f-b233-ddbcbe0f9821)



**Technologies Used**

TensorFlow & Keras: Deep learning framework used to build and train the fruit classification model using Convolutional Neural Networks (CNN).
Flask: A Python-based web framework used to develop the user interface and handle the backend.
OpenCV: A computer vision library used for image processing, such as edge detection and feature extraction.
scikit-image: Used for advanced image processing tasks, including texture analysis.
Pandas: A data manipulation library used to manage user feedback and classification results.


**Project Structure**

**The project directory is organized as follows:**

bash
Copy
/Product-Quality-Detection
|-- /static
| |-- /css
| | |-- style.css # Styles for the web interface
| |-- /images
| | |-- breda_robotics.png # Logos
| | |-- utrecht_university.png
|
|-- /templates
| |-- index.html # Homepage for the web application
| |-- result.html # Page displaying the analysis result
| |-- dashboard.html # Dashboard for displaying user feedback statistics
| |-- info.html # Information pages about the company and university
|
|-- /uploads # Folder where uploaded images are stored
|-- /models # Folder containing trained models
| |-- fruit_classifier.h5 # Pretrained fruit classification model
|
|-- /results # Folder for saving feedback data
| |-- user_feedback.csv # CSV file storing feedback from users
|
|-- app.py # Flask application and backend logic
|-- requirements.txt # List of dependencies for the project
|-- README.md # Project description and instructions

**Setup and Installation**

Clone the Repository:

First, clone the repository to your local machine:

bash
Copy
git clone https://github.com/yourusername/Product-Quality-Detection.git
cd Product-Quality-Detection
Install Dependencies:

Make sure you have Python 3 installed on your system. Install the required libraries using pip:

bash
Copy
pip install -r requirements.txt
Download the Pretrained Model:

Ensure that the pretrained model file (fruit_classifier.h5) is placed in the models folder. If not, you can train your own model by running the training script provided in the repository.

Run the Flask Application:

To start the web application, run the following command:

bash
Copy
python app.py
This will start the Flask server on localhost (default port: 5000).

Access the Web Interface:

Open your browser and go to:

arduino
Copy
http://127.0.0.1:5000/
You can now upload fruit images or use your camera to classify fruits.


**How It Works**

Image Upload/Camera Input:

Users can upload an image of a fruit, or use their device’s camera for real-time analysis.
The image is processed to extract important features such as texture, color, and edges.
Feature Extraction:

The system extracts various features from the image, including texture features (using GLCM), edge detection, and color analysis.
These features are fed into a Convolutional Neural Network (CNN) model, which has been trained to recognize different types of fruits and assess their quality.
Fruit Classification and Quality Detection:

The model predicts the type of fruit (e.g., Apple, Banana, etc.) and its quality (Fresh, Medium, Rotten).
The results are displayed to the user along with an option to provide feedback on the classification.
User Feedback:

Users are asked if they agree with the classification result. If they disagree, they can select the correct fruit and quality from the dropdown menu.
This feedback is stored in a CSV file for future use, helping to retrain and improve the model over time.
Real-Time Dashboard:

The system also provides a dashboard where statistics on fruit classifications and user feedback can be visualized.

**Future Improvements**
Model Expansion: Extend the model to support more fruit types and quality categories.
Low-Quality Image Handling: Enhance the model's ability to handle low-quality images or images with poor lighting.
Integration with Real-Time Applications: Deploy the system for use in retail or agricultural environments where fruits can be classified in real-time as they enter the market.

**Contributing**
We welcome contributions! If you have any ideas or improvements to make, feel free to fork the repository, make your changes, and create a pull request.

License
This project is licensed under the MIT License.

**Contact**
Developed by: Tareq Tuaayman
Master in Next Level Engineering - Utrecht University of Applied Sciences

**Additional Notes**
User Feedback: This feature helps improve the model's performance over time. As more feedback is gathered, the model’s accuracy and robustness will increase.
Real-Time Feedback: By implementing feedback loops, the system is capable of adapting to new types of fruits and different quality categories.
Final Notes
Feel free to ask any questions or report issues using the Issues section on the GitHub repository page. Thank you for using the AI-Based Fruit Classification System!
