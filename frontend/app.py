
import os
import sys
import streamlit as st


# =========================================================
# 1. PROJECT PATH
# =========================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# =========================================================
# 2. SESSION STATE
# =========================================================

if "liveness_result" not in st.session_state:
    st.session_state.liveness_result = None

if "passport_uploaded" not in st.session_state:
    st.session_state.passport_uploaded = False


# =========================================================
# 3. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Passpotron",
    page_icon="🛂",
    layout="centered"
)

st.title("🛂 Passpotron")
st.subheader("Enhanced Passport Verification System")

st.write(
    "Complete face liveness verification before uploading "
    "your passport."
)

st.divider()


# =========================================================
# 4. START LIVENESS VERIFICATION
# =========================================================

st.header("Step 1: Liveness Verification")

st.write(
    "Click the button below to start webcam verification. "
    "Follow the random challenge instructions shown in the "
    "webcam window. Press Q when you want to close the webcam."
)

if st.button("Start Liveness Verification"):
    # Clear the previous attempt's result
    st.session_state.liveness_result = None

    original_directory = os.getcwd()

    try:
        from anti_spoofing.combined_liveness import start_combined_webcam

        st.info("Starting webcam verification...")

        result = start_combined_webcam()

        st.session_state.liveness_result = result

    except Exception as e:
        st.session_state.liveness_result = "INCONCLUSIVE"
        st.error("Verification could not be completed.")
        st.exception(e)

    finally:
        os.chdir(original_directory)


# =========================================================
# 5. DISPLAY LIVENESS RESULT
# =========================================================

if st.session_state.liveness_result == "LIVE CANDIDATE":

    st.success("Liveness verification passed!")

    st.write(
        "Your liveness verification is successful. "
        "You can now proceed to passport upload."
    )

elif st.session_state.liveness_result == "SPOOF SUSPECTED":

    st.error("Spoof suspected!")

    st.warning(
        "Passport upload is locked because spoofing was suspected."
    )

elif st.session_state.liveness_result == "INCONCLUSIVE":

    st.warning("Liveness verification was inconclusive.")

    st.info(
        "Passport upload remains locked. "
        "You can start verification again."
    )

elif st.session_state.liveness_result is None:

    st.info(
        "Liveness verification has not been completed yet."
    )


st.divider()


# =========================================================
# 6. PASSPORT UPLOAD
# =========================================================

st.header("Step 2: Passport Upload")

if st.session_state.liveness_result == "LIVE CANDIDATE":

    st.success("Passport upload unlocked!")

    uploaded_file = st.file_uploader(
        "Upload your passport image or PDF",
        type=["jpg", "jpeg", "png", "pdf"]
    )

    if uploaded_file is not None:

        st.write("Uploaded file:", uploaded_file.name)

        if uploaded_file.type.startswith("image/"):

            st.image(
                uploaded_file,
                caption="Uploaded Passport",
                use_container_width=True
            )

        elif uploaded_file.type == "application/pdf":

            st.success("Passport PDF received successfully.")

        st.session_state.passport_uploaded = True

else:

    st.warning(
        "Passport upload is locked. "
        "Please pass liveness verification first."
    )
