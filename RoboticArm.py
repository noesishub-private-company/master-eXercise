import streamlit as st
from huggingface_hub import InferenceClient, HfApi
import os
from pydantic import BaseModel
import openai
import json
import requests
import datetime

# Set page config and compact title
st.set_page_config(layout="wide", page_title="eXercise")
# st.markdown("<h1 style='text-align: center;'>eXercise</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
/* Increase font size of tab labels */
[data-testid="stTabs"] button[role="tab"] {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# Tabs
main_tab, advisor_tab = st.tabs(["Robotic Arm Selector", "AI Advisor"])

with main_tab:
    col1, col2, col3 = st.columns([1, 0.5, 0.5])

    with col1:
        selected_model = st.selectbox("Select the model to use:", [
            "Qwen/Qwen2.5-72B-Instruct", 
            "HuggingFaceH4/zephyr-7b-beta", 
            "mistralai/Mistral-7B-Instruct-v0.3",
            "gpt-4o",
            "gpt-4o-mini"
        ], key="model_used")

        number_of_arms = st.number_input("Number of Robotic Arms", key="NOF_ROBOTIC_ARMS", value=5)

        initial_system_instructions = "You are an LLM Algorithm that will facilitate the implementation of a Language-to-XR Scene Component..."

        if "llm_messages" not in st.session_state:
            st.session_state.llm_messages = []
            st.session_state.llm_messages.append({"role": "system", "content": initial_system_instructions})
            first_llm_message = "Please provide your inquiry in natural language and the distances of the robotic arms..."
            st.session_state.llm_messages.append({"role": "assistant", "content": first_llm_message})

        if "response_llm" not in st.session_state:
            st.session_state["response_llm"] = ""

        chat_display_screen = st.container(height=350)
        chat_display_screen.chat_message("ai", avatar=":material/precision_manufacturing:").markdown(st.session_state.llm_messages[-1]["content"])

        if user_prompt := st.chat_input("Enter your prompt here"):
            if len(user_prompt) < 20:
                st.error("Please enter a prompt with more than 20 characters")
            else:
                chat_display_screen.chat_message("user", avatar=":material/engineering:").markdown(user_prompt)

                prompt_to_llm = f"The trainer said: {user_prompt} The distances of the robotic arms are: "
                prompt_to_llm += ", ".join([f"Robotic Arm Number {i+1}, Distance = {st.session_state[f'distance-{i}']}" for i in range(st.session_state['NOF_ROBOTIC_ARMS'])])
                prompt_to_llm += f". Use the information provided previously to suggest which robotic arm number (1 to {st.session_state['NOF_ROBOTIC_ARMS']}) aligns with that information..."
                prompt_to_llm += f"Remember, the trainer said: {user_prompt}"

                class RoboticArmRecommendation(BaseModel):
                    Robotic_Arm_Virtual_Fire: int
                    Robotic_Arm_Malfunctioning: int
                    Reason_of_selection: str

                st.session_state.llm_messages.append({"role": "user", "content": prompt_to_llm})
                st.session_state["response_llm"] = ""

                if st.session_state["model_used"].startswith("gpt-"):
                    st.session_state.client = openai.OpenAI()
                    client_response_llm = st.session_state.client.beta.chat.completions.create(
                        messages=st.session_state.llm_messages,
                        model=st.session_state["model_used"],
                        max_tokens=512
                    )
                    st.session_state["response_llm"] = client_response_llm.choices[0].message.content
                else:
                    llm_client = InferenceClient(model=st.session_state["model_used"], token=os.getenv("HF_TOKEN"))
                    st.session_state["response_llm"] = llm_client.text_generation(
                        prompt_to_llm,
                        grammar={"type": "json", "value": RoboticArmRecommendation.model_json_schema()},
                        max_new_tokens=250,
                        temperature=1.6,
                        return_full_text=False,
                    )

                try:
                    response_llm_json = json.loads(st.session_state["response_llm"])
                    str_display = "Robotic Arm Virtual Fire: " + str(response_llm_json["Robotic_Arm_Virtual_Fire"])
                    str_display += "\nRobotic Arm Malfunctioning: " + str(response_llm_json["Robotic_Arm_Malfunctioning"])
                    str_display += "\nReason of selection: " + response_llm_json["Reason_of_selection"]
                except:
                    str_display = st.session_state["response_llm"]

                chat_display_screen.chat_message("ai", avatar=":material/precision_manufacturing:").markdown(str_display)
                st.session_state.llm_messages.append({"role": "assistant", "content": st.session_state["response_llm"]})

                with open("llm_messages.json", "w") as json_file:
                    json.dump(st.session_state["llm_messages"], json_file, indent=4)

                with open("response_llm_json.json", "w") as json_file:
                    json.dump(response_llm_json, json_file, indent=4)

                api = HfApi()
                with open("response_llm_json.json", "rb") as fobj:
                    api.upload_file(
                        path_or_fileobj=fobj,
                        path_in_repo="response_llm_json.json",
                        repo_id="noesishub/XYZ",
                        repo_type="dataset",
                        commit_message="Upload generated file",
                        token=os.getenv("HF_TOKEN")
                    )

        try:
            st.json(st.session_state["llm_messages"], expanded=False)
        except:
            st.write(st.session_state["llm_messages"])

        st.write("Sample prompts")
        st.json({
            "scenario_1": "There is a virtual fire close to the base...",
            "scenario_2": "A virtual fire is detected at a far distance...",
        }, expanded=False)

    with col2:
        st.header("Distance Input", help="The trainee is the point of reference...")
        st.write("(zero distance is the point where the trainee is located - working)")

        import random
        if "initial_distances" not in st.session_state:
            st.session_state["initial_distances"] = [random.uniform(1, 10) for _ in range(100)]

        for i in range(st.session_state["NOF_ROBOTIC_ARMS"]):
            st.number_input(f"Enter distance of **robotic arm {i+1}**", 
                            value=st.session_state["initial_distances"][i], 
                            format="%.2f", step=0.01, key=f"distance-{i}")

    with col3:
        st.header("AI Recommendation")
        if st.session_state["response_llm"] != "":
            try:
                st.json(st.session_state["response_llm"])
            except:
                st.write(st.session_state["response_llm"])
        st.markdown("---")
        st.write("Selected AI Model: " + st.session_state["model_used"])
        st.markdown("*LLMs may generate inaccurate responses; please verify...*")


with advisor_tab:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        with st.form("advisor_form"):
            st.subheader("Required Inputs")

            with st.expander("AWS Configuration", expanded=False):
                aws_access_key_id = st.text_input("AWS Access Key ID")
                aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password")
                col_aws1, col_aws2 = st.columns(2)
                with col_aws1:
                    aws_region_name = st.selectbox("Region", [
                        "us-east-1", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1",
                        "ap-southeast-1", "ap-northeast-1", "ap-south-1"
                    ])
                with col_aws2:
                    timezone = st.selectbox("Timezone", [
                        "UTC", "Europe/Athens", "Europe/Berlin", "America/New_York", "Asia/Tokyo", "Australia/Sydney"
                    ])
                s3_bucket = st.text_input("S3 Bucket Name")
                prefix = st.text_input("S3 Prefix (e.g. v2/.../participant_data/)")
                participant_id = st.text_input("Participant Full ID")

            with st.expander("Baseline Phase Settings", expanded=False):
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    baseline_date = st.date_input("Baseline Date", value=datetime.date(2025, 2, 27))
                with col_b2:
                    baseline_start_time = st.time_input("Start Time", value=datetime.time(21, 0))
                baseline_end_time = st.time_input("End Time", value=datetime.time(21, 30))

            with st.expander("Classification Phase Settings", expanded=False):
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    classification_date = st.date_input("Classification Date", value=datetime.date(2025, 2, 27))
                with col_c2:
                    classification_start_time = st.time_input("Start Time", value=datetime.time(12, 0))
                classification_end_time = st.time_input("End Time", value=datetime.time(23, 0))

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    initiation_phase_duration_min = st.number_input("Initiation Duration (min)", value=5)
                with col_d2:
                    finish_line_phase_duration_min = st.number_input("Finish-Line Duration (min)", value=5)

            with st.expander("Model & API Settings", expanded=False):
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    llm_model = st.selectbox("LLM Model", ["deepseek-chat", "gpt-4o", "mistralai/Mistral-7B-Instruct-v0.3"])
                with col_m2:
                    llm_api_key = st.text_input("LLM API Key", type="password")
                api_url = st.text_input("API Endpoint", value="https://exrercise.8bellsresearch.com/classify_stress/")

            # Check required fields
            required_fields = [
                aws_access_key_id, aws_secret_access_key, aws_region_name, s3_bucket,
                prefix, participant_id, timezone, llm_api_key, llm_model, api_url
            ]
            if all(field.strip() != "" for field in required_fields):
                st.success("All required fields are filled.")
            else:
                st.warning("Please complete all required fields before submitting.")

            submit_clicked = st.form_submit_button("Get AI Advisor Recommendation")

    # Prepare params after form
    params = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_region_name": aws_region_name,
        "s3_bucket": s3_bucket,
        "prefix": prefix,
        "participant_id": participant_id,
        "timezone": timezone,
        "baseline_date": baseline_date.strftime("%Y-%m-%d"),
        "baseline_start_time": baseline_start_time.strftime("%H:%M"),
        "baseline_end_time": baseline_end_time.strftime("%H:%M"),
        "classification_date": classification_date.strftime("%Y-%m-%d"),
        "classification_start_time": classification_start_time.strftime("%H:%M"),
        "classification_end_time": classification_end_time.strftime("%H:%M"),
        "initiation_phase_duration_min": initiation_phase_duration_min,
        "finish_line_phase_duration_min": finish_line_phase_duration_min,
        "llm_api_key": llm_api_key,
        "llm_model": llm_model
    }

    with right_col:
        if submit_clicked:
            with st.spinner("Contacting the API..."):
                try:
                    resp = requests.get(api_url, params=params, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()

                    advisor = data.get("ai_advisor_recommendation", None)
                    if advisor:
                        st.success("AI Advisor Recommendation:\n\n" + advisor)
                    else:
                        st.warning("No AI Advisor Recommendation returned from API.")

                    with st.expander("Full API response"):
                        st.json(data)
                except Exception as e:
                    st.error(f"API call failed: {e}")
