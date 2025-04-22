import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr
from PIL import Image

def detect_plate_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return img, "Plate not found"

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1), (x2, y2) = (np.min(x), np.min(y)), (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(cropped_image)

    text = result[0][-2] if result else "Text not found"

    cv2.putText(img, text, (location[0][0][0], location[1][0][1]+60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)

    return img, text

# Streamlit UI
st.title("🔍 License Plate Reader")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Detecting license plate..."):
        processed_image, plate_text = detect_plate_from_image(image_np)

    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Detected License Plate", use_column_width=True)
    st.success(f"Detected Plate: {plate_text}")
