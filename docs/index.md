# eXRercise sub-project of the MASTER first Open Call (OC)


The following documentation addresses the integration of assets developed using Unity Engine and the Unity Editor, for the purposes of the MASTER OC project **eXercise**.

One of the components of the **eXercise Training System** is the creation of VR-based scenarios. The library described below enables dynamic VR scenarios for:
- fire incidents (by creating a virtual fire effect at a robotic arm/device)
- malfunction scenarios at a specified robotic arm or device

This component accepts structured decisions output from an LLM prompt, used by a trainer. The result is a VR scenario presented to a trainee. It is distributed as a Unity-native DLL library.