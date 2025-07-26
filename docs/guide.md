# eXRercise Training System: Integration Guide

Welcome to the documentation for the **eXRercise Training System**, a pilot developed under the MASTER OC initiative. This guide walks you through how to:

- Set up the Unity DLL-based VR scenario system
- Connect and control it using a Streamlit UI with LLMs
- Deploy integrated XR training simulations with biometric feedback

> This documentation assumes you know your way around Unity, Python, and Hugging Face/Streamlit.

---

# 1. Setting Up the Unity Plugin

## Overview

The Unity module delivers interactive XR scenarios using robotic arms. It uses two core components:

- `BusinessLogic`: Controls animations, fires, and malfunctions.
- `HuggingFaceJsonFetcher`: Reads JSON scenario data from Hugging Face.

## 1.1 Import the DLL

1. Copy the distributed `.dll` file into `Assets/Plugins` in Unity.
2. Attach the `BusinessLogic` and `HuggingFaceJsonFetcher` scripts to a GameObject in your scene.

## 1.2 Configure BusinessLogic

Use the Unity Inspector to set the following:

| Field | Description |
|-------|-------------|
| `deviceAnimators` | Animator components for each robotic arm |
| `fireInstances` | Pool of fire prefabs |
| `fireOffset` | Offset vector (default: 0,1,0) |
| `alarmClip` | Audio for fire events |
| `fireAudioSource` | Source that plays fire audio |
| `malfunctioningClip` | Audio for malfunctions |
| `malfuntioningAudioSource` | Source for malfunction audio |

## 1.3 Configure HuggingFaceJsonFetcher

Set the following fields in the Inspector:

| Field | Description |
|-------|-------------|
| `repoId` | e.g., `noesishub/XYZ` |
| `filename` | JSON file (e.g., `response_llm_json.json`) |
| `token` | Hugging Face API token |

## 1.4 Sample JSON Format

```json
{
  "Robotic_Arm_Virtual_Fire": 1,
  "Robotic_Arm_Malfunctioning": 2,
  "Reason_of_selection": "some explanation here."
}
```

The `BusinessLogic` component reads this JSON and dynamically triggers visual and audio events accordingly.

---

# 2. Streamlit UI Setup for Trainer Control

## Overview

This interface allows trainers to define XR scenarios using natural language prompts and control Unity behavior indirectly via JSON files written to Hugging Face.

## 2.1 Interface Structure

The app contains:

- **Robotic Arm Selector Tab**: Prompt-based scenario generator
- **AI Advisor Tab**: Performance and stress analysis

## 2.2 Robotic Arm Selector

Functionality includes:

- Prompt input (e.g., "Set fire to closest robotic arm")
- Distance data for robotic arms
- LLM selection (OpenAI / Hugging Face)
- Output: JSON file uploaded to Hugging Face

> The prompt must describe with at least 20 characters the desired scenario. The trainer can either explicitely ask for only a fire or only a malfunctioning effect, or a combination. 

### Sample Output:

```json
{
  "Robotic_Arm_Virtual_Fire": 1,
  "Robotic_Arm_Malfunctioning": 3,
  "Reason_of_selection": "the explanation here for the selection"
}
```

> This file must match the Unity-side `filename` in HuggingFaceJsonFetcher!
>
> The numeric values can range from 1 to the actual number of the robotic arms. If the value is set to -1, no arm is selected by the LLM.

## 2.3 AI Advisor

Provides advanced analytics:

- Retrieves biometric stress data (from Empatica devices)
- Classifies session difficulty using CNN models
- Generates markdown reports via LLMs

## 2.4 Required Configuration

| Param | Description |
|-------|-------------|
| `aws_region_name` | AWS region for S3 |
| `s3_bucket` | Data source bucket |
| `participant_id` | Identifier |
| `baseline_date`, `start_time`, etc. | Time frame for analysis |

## 2.5 Notes

- Prompts shorter than 20 characters are disallowed.
- The system supports retry logic on JSON parsing failures.

---

# 3. Conclusion

The eXRercise system demonstrates the integration of Unity VR capabilities with LLM-driven training scenarios and biometric analysis. Follow the guide above to deploy and test your own scenarios using this hybrid XR + AI training platform.