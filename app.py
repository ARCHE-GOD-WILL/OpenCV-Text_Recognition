import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os
import sys

# Add current directory to path so we can import ocr_toolkit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_toolkit.pipeline import OCRPipeline, PipelineConfig

st.set_page_config(page_title="OpenCV OCR Tool", layout="wide")

st.title("OpenCV Text Recognition & Table Extraction (Batch Mode)")
st.markdown("Upload **one or more images** to extract text and tables. Results will be consolidated.")

# Sidebar Controls
st.sidebar.header("Configuration")

# Image Preprocessing
st.sidebar.subheader("1. Preprocessing")
remove_red = st.sidebar.checkbox("Remove Red (Stamp Removal)", value=True, help="Removes red stamps/seals that might interfere with text.")
sharpen = st.sidebar.checkbox("Sharpen", value=True)
denoise = st.sidebar.checkbox("Denoise", value=True)
correct_skew_opt = st.sidebar.checkbox("Correct Skew", value=True, help="Rotates image to align text horizontally.")

# Line Detection
st.sidebar.subheader("2. Line Detection (Table Structure)")
st.sidebar.info("Adjust these if the table grid is not detected correctly.")
h_canny_low = st.sidebar.slider("H Canny Low", 0, 255, 50)
h_canny_high = st.sidebar.slider("H Canny High", 0, 255, 150)
h_hough_threshold = st.sidebar.slider("H Hough Threshold", 10, 300, 150, help="Lower this to detect shorter/fainter horizontal lines.")
h_min_line_length = st.sidebar.slider("H Min Line Length", 10, 500, 100)

v_canny_low = st.sidebar.slider("V Canny Low", 0, 255, 20)
v_canny_high = st.sidebar.slider("V Canny High", 0, 255, 40)
v_hough_threshold = st.sidebar.slider("V Hough Threshold", 10, 300, 10, help="Lower this to detect shorter/fainter vertical lines.")

min_gap = st.sidebar.slider("Min Gap (Merge Lines)", 0, 100, 40, help="Merge lines that are close together.")

# OCR
st.sidebar.subheader("3. OCR Settings")
lang = st.sidebar.text_input("Language", "chi_sim+eng")
psm = st.sidebar.selectbox(
    "Page Segmentation Mode", 
    [3, 4, 6, 7, 11, 12], 
    index=2, 
    format_func=lambda x: f"PSM {x} - " + {
        3: "Fully automatic page segmentation, but no OSD.",
        4: "Assume a single column of text of variable sizes.",
        6: "Assume a single uniform block of text.",
        7: "Treat the image as a single text line.",
        11: "Sparse text. Find as much text as possible in no particular order.",
        12: "Sparse text with OSD."
    }.get(x, "Unknown")
)

# Allow multiple file uploads
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Prepare consolidated storage
    all_dataframes = []
    
    # Process each file
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Configuration object (same for all files in batch)
    config = PipelineConfig(
        remove_red=remove_red,
        sharpen=sharpen,
        denoise=denoise,
        correct_skew=correct_skew_opt,
        h_canny_low=h_canny_low,
        h_canny_high=h_canny_high,
        h_hough_threshold=h_hough_threshold,
        h_min_line_length=h_min_line_length,
        v_canny_low=v_canny_low,
        v_canny_high=v_canny_high,
        v_hough_threshold=v_hough_threshold,
        line_min_gap=min_gap,
        lang=lang,
        ocr_config=f"--psm {psm}"
    )
    pipeline = OCRPipeline(config)

    # Tabs for different views
    tab_overview, tab_details = st.tabs(["Consolidated Results", "Per-Image Details"])

    with tab_details:
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
            
            # Save temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
            tfile.write(uploaded_file.read())
            tfile.close()

            try:
                result = pipeline.run(tfile.name)
                
                with st.expander(f"Details: {uploaded_file.name}", expanded=(len(uploaded_files)==1)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(result.original_image, caption=f"Original: {uploaded_file.name}", channels="BGR", use_column_width=True)
                        if result.detected_lines:
                             # Draw lines on a copy for visualization
                            debug_img = result.corrected_image.copy() if result.corrected_image is not None else result.preprocessed_image.copy()
                            if len(debug_img.shape) == 2:
                                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
                            for y in result.detected_lines.horizontal:
                                cv2.line(debug_img, (0, y), (debug_img.shape[1], y), (0, 0, 255), 2)
                            for x in result.detected_lines.vertical:
                                cv2.line(debug_img, (x, 0), (x, debug_img.shape[0]), (0, 255, 0), 2)
                            st.image(debug_img, caption="Detected Structure", channels="BGR", use_column_width=True)

                    with col2:
                        if result.has_table:
                            df = result.to_dataframe()
                            # Add filename column for consolidation
                            df.insert(0, "Source File", uploaded_file.name)
                            all_dataframes.append(df)
                            st.dataframe(df)
                        else:
                            st.warning("No table structure detected.")
                            st.text_area("Raw Text", result.full_text, height=150)
                            # Create a simple DF for the raw text if needed, or just skip
                            # For now, we'll create a single-cell DF so it appears in consolidation
                            df_fallback = pd.DataFrame({"Source File": [uploaded_file.name], "Raw Text": [result.full_text]})
                            all_dataframes.append(df_fallback)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(tfile.name):
                    os.unlink(tfile.name)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("Processing complete!")
    progress_bar.empty()

    # Consolidated View
    with tab_overview:
        st.subheader("Batch Results")
        if all_dataframes:
            # Consolidate
            try:
                # We need to align columns if they differ, or just concat and fill NA
                consolidated_df = pd.concat(all_dataframes, ignore_index=True)
                
                st.success(f"Successfully processed {len(uploaded_files)} files. Total rows: {len(consolidated_df)}")
                
                # Interactive Editor
                edited_consolidated_df = st.data_editor(consolidated_df, num_rows="dynamic", use_container_width=True)
                
                # Download Button
                csv = edited_consolidated_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Consolidated CSV",
                    data=csv,
                    file_name="batch_ocr_results.csv",
                    mime="text/csv",
                    key='download-consolidated-csv'
                )
            except Exception as e:
                st.error(f"Error consolidating results: {e}")
        else:
            st.info("No data extracted from uploaded files.")
