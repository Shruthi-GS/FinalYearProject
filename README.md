BRAILLE DETECTION AND AUDIO OUTPUT FOR INCLUSIVE COMMUNICATION

Bridging the Gap Between Tactile and Digital Communication


MOTIVATION:

To address communication barriers faced by visually impaired individuals in accessing written content, help normal-sighted individuals to access Braille content and overcome the limitations of manual Braille translation.

OBJECTIVES:

1-Automate Braille-to-text conversion using image processing.

2-Integrate text-to-speech functionality for accessibility.

3-Provide a user-friendly web interface.

4-Promote inclusivity through accurate text and audio outputs.

METHODOLOGY:

Image Pre-processing: Noise reduction using Gaussian blur, thresholding and erosion.

Gaussian blur
![image](https://github.com/user-attachments/assets/09e63a47-2a64-4584-b91c-88d001616840)

Erosion

![image](https://github.com/user-attachments/assets/8bc4b1e2-5ccf-478e-86a2-f9183354de63)

![image](https://github.com/user-attachments/assets/6420b702-6df6-4742-a537-f2569d46d656)


Connected Component Labelling: Apply CCL to isolate Braille dots.

Feature Extraction: Apply bounding box around each character.
![image](https://github.com/user-attachments/assets/d4d0b3de-0ac7-4194-bb27-5bd8de9f505f)

Mapping and Conversion: Use of a JSON key for Braille-to-text translation.
![image](https://github.com/user-attachments/assets/9c75f8ae-16a8-4863-bf2c-c43119359d06)

Text-to-Speech Integration: gTTS for converting text to audio.
Technologies Used: OpenCV, NumPy, Django, HTML, CSS, JavaScript, jQuery.




TECH STACK: OpenCV, NumPy, Django, HTML, CSS, JavaScript, jQuery.
