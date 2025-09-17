import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

from license_detector import LicensePlateRecognizer

@st.cache_resource
def load_recognizer():
    """Load the recognizer with caching"""
    return LicensePlateRecognizer(languages=['en'], gpu=False)

def main():
    st.set_page_config(
        page_title="Modern License Plate Recognition",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Modern License Plate Recognition")
    st.markdown("**Powered by OpenCV + EasyOCR** - No training required!")
    
    # Load recognizer
    with st.spinner("Loading recognition system..."):
        try:
            recognizer = load_recognizer()
            st.success("✅ System ready!")
        except Exception as e:
            st.error(f"❌ Failed to load system: {e}")
            st.stop()
    
    # Sidebar options
    with st.sidebar:
        st.header("⚙️ Settings")
        
        show_debug = st.checkbox("Show debug info", value=False)
        show_all_results = st.checkbox("Show all detections", value=False)
        
        st.header("📊 About")
        st.info("""
        **This system uses:**
        - OpenCV for plate detection
        - EasyOCR for text recognition
        - No training data required!
        - Works with multiple languages
        """)
    
    # Main interface
    uploaded_file = st.file_uploader(
        "📤 Upload an image with a license plate",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image containing a license plate"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Process button
        if st.button("🚀 Recognize License Plate", type="primary"):
            with st.spinner("Processing..."):
                # Convert PIL to OpenCV format
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    cv_image = image_array
                
                # Run recognition
                results = recognizer.recognize_from_image(cv_image)
                
                with col2:
                    st.subheader("🎯 Results")
                    
                    if 'error' in results:
                        st.error(f"❌ {results['error']}")
                    elif results['best_result']:
                        best = results['best_result']
                        
                        # Main result
                        st.success(f"**License Plate: {best['text']}**")
                        st.info(f"**Confidence: {best['confidence']:.1%}**")
                        
                        # Confidence bar
                        st.progress(min(best['confidence'], 1.0))
                        
                        # Method used
                        st.caption(f"Detection method: {best['method']}")
                        
                        # Show all results if requested
                        if show_all_results and len(results['plates']) > 1:
                            st.subheader("🔍 All Detections")
                            for i, plate in enumerate(results['plates']):
                                st.write(f"{i+1}. **{plate['text']}** (confidence: {plate['confidence']:.1%}, method: {plate['method']})")
                    
                    else:
                        st.warning("⚠️ No license plates detected")
                        st.info("💡 Try with a clearer image or different angle")
                
                # Debug information
                if show_debug:
                    with st.expander("🔧 Debug Information"):
                        for info in results.get('processing_info', []):
                            st.write(info)
                        
                        if 'plates' in results:
                            st.write("**All detections:**")
                            st.json(results['plates'])
                
                # Create visualization
                try:
                    # Save visualization
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        recognizer.visualize_results(cv_image, results, tmp_file.name)
                        
                        # Show visualization
                        viz_image = Image.open(tmp_file.name)
                        st.subheader("📊 Detection Visualization")
                        st.image(viz_image, caption="Detected Regions", use_column_width=True)
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    if show_debug:
                        st.error(f"Visualization error: {e}")

    # Usage instructions
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        **For best results:**
        - Use clear, well-lit images
        - Ensure the license plate is not too small or too blurry
        - Avoid extreme angles or perspectives
        
        **This system automatically:**
        - Detects license plate regions
        - Extracts text using advanced OCR
        - Cleans and validates the results
        - No training data required!
        """)

if __name__ == "__main__":
    main()