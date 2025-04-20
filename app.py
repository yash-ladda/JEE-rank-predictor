# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load model and encoders
# model = joblib.load("compressed_log_transformed_model.pkl")  # Update path if needed
# mapping_set = joblib.load("encoders.pkl")  # Save this from your notebook

# # Custom Background and Styling
# st.markdown("""
#     <style>
#         .stApp {
#             background-image: linear-gradient(to bottom right, #ff4aff, #7affaa);
#             background-attachment: fixed;
#         }
#         h1 {
#             font-family: 'Helvetica Neue', sans-serif;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Sidebar - Project Info
# st.sidebar.title("Project Info")
# st.sidebar.markdown("**Project Title:** JEE College Closing Rank Predictor")
# st.sidebar.markdown("**Subject:** DWDA")
# st.sidebar.markdown("**College:** [Your College Name]")

# st.sidebar.markdown("** Contributors:**")
# st.sidebar.markdown("""
# - Yash Ladda  
# - Pritesh Mantri  
# - Mithilesh Dhoot  
# - Samiksha Deshpande  
#   SY CE (SE) A
# """)

# # Main Title and Description
# st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎯 JEE College Closing Rank Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<h4 style='text-align: center;'>Enter the details below to predict the expected closing rank for your preferred branch</h4>", unsafe_allow_html=True)
# st.write("")

# # Form for inputs
# with st.form("rank_predictor_form"):
#     year = st.number_input("📅 Enter Year", min_value=2016, max_value=2050, step=1)
#     round_no = st.selectbox("🔁 Round Number", [1, 2, 3, 4, 5, 6, 7])
#     quota = st.selectbox("🎫 Quota", ["AI", "HS", "OS"])
#     gender = st.selectbox("⚧️ Gender", ["Both", "Female"])
#     institute_name = st.selectbox("🏫 Institute", sorted(mapping_set['institute_name'].classes_))
#     branch = st.selectbox("💼 Branch", sorted(mapping_set['branch'].classes_))
#     category = st.selectbox("📚 Category", sorted(mapping_set['category'].classes_))

#     submitted = st.form_submit_button("🚀 Predict Closing Rank")

#     if submitted:
#         # Prepare input
#         input_data = pd.DataFrame({
#             'year': [year],
#             'round_no': [round_no],
#             'quota': [quota],
#             'gender': [gender],
#             'institute_name': [institute_name],
#             'branch': [branch],
#             'category': [category]
#         })

#         # Encode
#         for col in input_data.columns:
#             if col in mapping_set:
#                 le = mapping_set[col]
#                 input_data[col] = input_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

#         # Predict
#         log_rank = model.predict(input_data)
#         predicted_rank = np.expm1(log_rank)[0]

#         st.success(f"🎉 Predicted Closing Rank: **{int(predicted_rank)}**")

# # Footer
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; font-size: 12px;'>Made with ❤️ by Team SY CE (SE) A</p>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("compressed_log_transformed_model.pkl")
mapping_set = joblib.load("encoders.pkl")

# ---- Custom Styling ----
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://wallpaperaccess.com/full/317501.jpg');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }

        /* Change color of label/input text */
        label, .css-1cpxqw2, .css-qrbaxs, .css-1d391kg {
            color: black !important;
            font-weight: 600;
        }

        .title {
            font-size: 40px;
            text-align: center;
            color: #aaffff;
            text-shadow: 2px 2px 5px #000000;
            font-weight: 800;
        }

        .subtitle {
            font-size: 25px;
            text-align: center;
            color: black;
            font-weight: 600;
        }

        .prediction {
            background-color: rgba(0,0,0,0.8);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 20px;
            text-align: center;
        }

        .stButton > button {
            background-color: #00c853;
            color: white;
            font-size: 400px;
            border-radius: 10px;
            padding: 10px 20px;
        }

        .stButton > button:hover {
            background-color: #00b248;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.title("Project Info")
st.sidebar.markdown("**Project Title:** JEE College Closing Rank Predictor")
st.sidebar.markdown("**Subject:** DWDA")
st.sidebar.markdown("**College:** VIIT, Pune")
st.sidebar.markdown("**Contributors:**")
st.sidebar.markdown("""
- Yash Ladda  
- Pritesh Mantri  
- Mithilesh Dhoot  
- Samiksha Deshpande  
""")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: white;'>Made with ❤️ by Team SY CE (SE) A</p>", unsafe_allow_html=True)

# ---- Main Title ----
st.markdown("<div class='title'>JEE College Closing Rank Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter the details below to get your expected closing rank</div>", unsafe_allow_html=True)

# ---- Input Form ----
with st.form("form"):
    year = st.number_input("Enter Year", min_value=2016, max_value=2050, step=1)
    round_no = st.selectbox("Round Number", [1, 2, 3, 4, 5, 6, 7])
    quota = st.selectbox("Quota", ["AI", "HS", "OS"])
    gender = st.selectbox("Gender", ["Both", "Female"])
    institute_name = st.selectbox("Institute", sorted(mapping_set['institute_name'].classes_))
    branch = st.selectbox("Branch", sorted(mapping_set['branch'].classes_))
    category = st.selectbox("Category", sorted(mapping_set['category'].classes_))

    submitted = st.form_submit_button("Predict Closing Rank")

# ---- Prediction ----
if submitted:
    input_df = pd.DataFrame({
        'year': [year],
        'round_no': [round_no],
        'quota': [quota],
        'gender': [gender],
        'institute_name': [institute_name],
        'branch': [branch],
        'category': [category]
    })

    # Encode
    for col in input_df.columns:
        if col in mapping_set:
            le = mapping_set[col]
            input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Predict
    log_rank = model.predict(input_df)
    predicted_rank = np.expm1(log_rank)[0]

    st.markdown(f"<div class='prediction'>Predicted Closing Rank: <br><span style='font-size:32px;'>{int(predicted_rank)}</span></div>", unsafe_allow_html=True)

# ---- Footer ----

