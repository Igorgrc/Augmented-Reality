import numpy as np
import cv2
import utils
import streamlit as st

def main():
    st.title("ArUco Markers Processing")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    intrinsics, dist_coeffs = utils.calibrate_camera('./calibration/')
    if uploaded_file is not None:
        # Read image as OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)  # 1 for color

        # Show uploaded image in the app
        st.image(image, caption='Uploaded Image', channels="BGR")

        # Run your processing function
        utils.plot_image_in_marker(image, intrinsics, dist_coeffs)

if __name__ == '__main__':
    main()
