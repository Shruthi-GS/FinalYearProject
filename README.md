BRAILLE DETECTION AND AUDIO OUTPUT FOR INCLUSIVE COMMUNICATION

Bridging the Gap Between Tactile and Digital Communication


MOTIVATION:

To address communication barriers faced by visually impaired individuals in accessing written content, help normal-sighted individuals to access Braille content and overcome the limitations of manual Braille translation.

OBJECTIVES:

1- Automate Braille-to-text conversion using image processing.

2- Integrate text-to-speech functionality for accessibility.

3- Provide a user-friendly web interface.

4- Promote inclusivity through accurate text and audio outputs.

METHODOLOGY:

1- Image Pre-processing: Noise reduction using Gaussian blur, thresholding and erosion.

Gaussian blur

![image](https://github.com/user-attachments/assets/09e63a47-2a64-4584-b91c-88d001616840)

Erosion

![image](https://github.com/user-attachments/assets/8bc4b1e2-5ccf-478e-86a2-f9183354de63)

![image](https://github.com/user-attachments/assets/6420b702-6df6-4742-a537-f2569d46d656)


2- Connected Component Labelling: Apply CCL to isolate Braille dots.

3- Feature Extraction: Apply bounding box around each character.

![image](https://github.com/user-attachments/assets/d4d0b3de-0ac7-4194-bb27-5bd8de9f505f)

4- Mapping and Conversion: Use of a JSON key for Braille-to-text translation.

![image](https://github.com/user-attachments/assets/9c75f8ae-16a8-4863-bf2c-c43119359d06)

5- Text-to-Speech Integration: gTTS for converting text to audio.

TECH STACK: OpenCV, NumPy, Django, HTML, CSS, JavaScript, jQuery.

RESULTS:

1- Successfully converts Braille images into English text and audio.

2- User-friendly web interface for seamless interaction.

3- High accuracy in character recognition using preprocessing techniques.

4- Modular design ensures scalability and adaptability to other languages and devices.
![image](https://github.com/user-attachments/assets/7bfc84dd-2e92-47ce-aed1-2aabd0a1d8a9)
![image](https://github.com/user-attachments/assets/44f34807-a857-4b40-a8d9-13f1a4388c5d)
![image](https://github.com/user-attachments/assets/2a318ac6-8ad5-42b8-b744-ed64bcbb65b0)
![image](https://github.com/user-attachments/assets/603cab75-0aba-4fcc-93af-7227afc41731)

DISCUSSION:

The project successfully automates Braille-to-text and audio translation, addressing accessibility challenges for visually impaired individuals.
Advanced image processing and text-to-speech technologies ensure high accuracy and inclusivity.
A scalable and modular design makes the system adaptable for diverse languages and future enhancements.
Promotes independence and accessibility in education, communication, and daily life for the visually impaired.



FUTURE SCOPE:

Support for Grade 2 Braille and multilingual capabilities.
Mobile optimization and real-time processing.
Advanced recognition with Vision Transformers and Transfer Learning.


