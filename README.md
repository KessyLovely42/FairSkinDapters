# Mitigating Skin Tone Bias in Skin Cancer Detection: A Low Rank Adaptation Approach

This a prototype web application built using a microservice architecture, with model inference exposed through a dedicated FastAPI endpoint. Users upload a skin lesion image via the interface, which is then preprocessed and passed to an inference service powered by a pretrained Vision Transformer augmented with Low-Rank Adaptation (LoRA) adapters. The service returns a predicted class label along with a probability score indicating model confidence. To support transparency and trust, the system also generates attention visualisations that highlight the image regions most influential to the prediction. These outputs are returned to the client application, allowing users to inspect both the diagnostic result and its visual explanation.



![User interface](image.png)

User interface of the app. 

![Interface during prediction](image-1.png)
The above image shows the user interface during prediction. 

### Getting Started
- Install PythonEnsure that Python 3.9 or later is installed on your system. You can verify this by running `python --version` in your terminal.
- Create a Virtual Environment to isolate project dependencies by running `python -m venv venv`.
- Activate the environment using source `venv/bin/activate` on macOS or Linux, or `venv\Scripts\activate` on Windows.
- Install Project Dependencies: Upgrade pip and install all required packages using `pip install --upgrade pip` followed by `pip install -r requirements.txt`.
- Run the application using either `fastapi run main.py` or `uvicorn main:app --reload`.
- Open your web browser and navigate to http://127.0.0.1:8000 to use the application and upload images for inference.


### Demonstration

For demonstration purposes, sample images are provided in the sample_images directory. These images are drawn from the test set used in the study and are intended solely for testing the application. The folder includes both benign and malignant cases, with examples from each Fitzpatrick skin tone subgroup (FST I–II and FST V–VI) to illustrate model behaviour across skin tones.

### Disclaimer

This application is a research prototype and is not intended for use in real-world clinical settings. The associated study and system have not undergone clinical validation and should not be relied upon for medical decision-making. Furthermore, the images provided are subject to copyright restrictions and are included strictly for non-commercial research purposes, in accordance with the terms set by the dataset providers. Use of the images is permitted solely under the express conditions outlined by the DDI Dataset: https://ddi-dataset.github.io/.