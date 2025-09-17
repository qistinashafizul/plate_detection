# License Plate Detection System 🚗🔍

An OCR-based system to detect license plate numbers from images, powered by **EasyOCR**, **OpenCV**, and an interactive **Streamlit demo**.

## 📌 Features

- Upload an image via the Streamlit web app
- Automatically detects and extracts license plate text
- Regex filtering to validate license plate formats
- Visual feedback with bounding boxes drawn on the image

## 🛠️ Tech Stack

- **Python 3.x**
- **Streamlit** - Interactive demo UI
- **EasyOCR** - Deep learning OCR engine
- **OpenCV** - Image processing and handling

## 📂 Project Structure

```
plate-detection/
├── ui.py                    # Streamlit app
├── license_detector.py      # Core detection logic
├── requirements.txt         # Project dependencies
└── examples/               # Sample images for testing
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/license-plate-detector.git
cd license-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run ui.py
```

The app will be available at `http://localhost:8501`

## 🌐 Live Demo

Deploy this project for free on:
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/spaces)

## 📸 Usage Example

1. Upload an image containing a license plate
2. The system automatically detects and extracts the plate number
3. View the results with bounding boxes overlaid on the original image

## 🔮 Future Improvements

- **Enhanced Preprocessing**: Add denoising, binarization, and deskewing capabilities
- **Custom Model Training**: Fine-tune OCR model for specific regional license plate formats
- **API Development**: Deploy as a REST API service for broader integration
- **Multi-language Support**: Extend recognition to international license plate formats
- **Performance Optimization**: Implement caching and batch processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

If you encounter any issues or have questions, please [open an issue](https://github.com/your-username/license-plate-detector/issues) on GitHub.
