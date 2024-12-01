import streamlit as st
import requests

# App Title
st.title("VectorShield🛡️: Trade Compliance Screening using AI 🤖")

# Text Input
user_input = st.text_area(
    "Enter the text to screen for compliance:",
    height=200,  # Adjust the height as needed
    placeholder="Paste or type a large block of text here..."
)
# Button for Prediction
if st.button("Predict"):
    if user_input:
        # Replace the following line with your prediction logic
        user_query = user_input

        url = "http://127.0.0.1:8000/predict"  # Replace with your server URL if deployed

        # Define the JSON payload
        payload = {
            "user_query": user_query
        }

        try:
            # Make the POST request
            response = requests.post(url, json=payload).json()

            #Extract response details
            label = response['response']['prediction']
            probability_score = response['response']['probability_score']
            explanation = f"""{response['response']['explain']}"""

            # label = "not sensitive"
            # probability_score = 1.0
            # explanation = "**This is a gun.**"

            # Display the prediction
            prediction = f"Prediction result for: {user_input}"

            # Display label with conditional coloring
            if label == "sensitive":
                label_text = "Sensitive"
                st.markdown(
                    f"<span style='color:red; font-weight:bold;'>Label: {label_text}</span>", 
                    unsafe_allow_html=True
                )
            elif label == "not_sensitive":
                label_text = "Not Sensitive"
                st.markdown(
                    f"<span style='color:green; font-weight:bold;'>Label: {label_text}</span>", 
                    unsafe_allow_html=True
                )
            
            # Additional information
            st.write(f"Probability Score: {probability_score}")
            # Display explanation inside a markdown
            st.header("AI Explanation 📝:")
            # Using a column layout to give more width to the text display

            st.text(explanation)
            # text_list = explanation.split("\n")

            # for text in text_list:
            #     st.markdown(text)


                

        except Exception as e:
            st.error("Error: Could not fetch prediction. Please try again.")
            st.write(f"Details: {e}")
    else:
        st.error("Please enter a valid text to proceed.")
