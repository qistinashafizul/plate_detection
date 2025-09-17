# License Plate Detection System 🚗🔍

An OCR-based system to detect license plate numbers from images, powered by **EasyOCR**, **OpenCV**, and an interactive **Streamlit demo**.

## 📌 Features
- Upload an image via the Streamlit web app
- Automatically detects and extracts license plate text
- Regex filtering to validate license plate formats
- Visual feedback with bounding boxes drawn on the image

## 🛠️ Tech Stack
- Python 3.x
- Streamlit (demo UI)
- EasyOCR (deep learning OCR)
- OpenCV (image handling)

## 📂 Project Structure
license-plate-detector/
│── app.py # Streamlit app
│── license_detector.py # detection logic
│── requirements.txt # dependencies
│── examples/ # sample images


## 🚀 Run Locally
Clone the repo and install dependencies:

git clone https://github.com/your-username/license-plate-detector.git
cd license-plate-detector

pip install -r requirements.txt
streamlit run app.py

## 🌐 Demo

👉 You can deploy this project for free on Streamlit Cloud
 or Hugging Face Spaces
.

## 📸 Example

Input image → Detected license plate:

## 🔮 Future Improvements

Preprocessing pipeline (denoising, binarization, deskewing)

Train/fine-tune OCR model for regional license plates

Deploy as a full web service with API

## 📜 License

This project is licensed under the MIT License.
