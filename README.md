# Emotion Recognition in Dialogue Agents with an Integrated Memory Module
Imperial Msc Computing Individual Project
Recent advancements in AI and Natural Language Processing have underscored the need for emotionally intelligent dialogue agents. This repository houses the code and resources pertaining to our efforts to bridge the gap between AI capabilities and emotionally charged human interactions.

## Overview
**Motivation:** The AI community acknowledges the significance of memory management, relational structures in dialogues, commonsense knowledge integration, emotion recognition, and multi-language capabilities. These insights aim to make dialogue agents more relatable and efficient.
**Aim:** To integratively enhance AI conversational systems using a Memory Module with established large language models. Our work focuses on making these systems adept at classifying human emotions.
**Tools & Datasets:** Leveraging tools like DialoGPT, OPT, and PyTorch, we've fine-tuned our models on the EmpatheticDialogues dataset to create a dialogue system that continually updates the conversational context based on emotion.
**Features**
Advanced Memory Module integration with DialoGPT and OPT.
Fine-tuning capabilities using the EmpatheticDialogues dataset.
Custom model training options including learning rate optimization, selective layer freezing, and batch size adjustments.
Enhanced emotion recognition from dialogues, improving the efficacy of AI-human interactions.

<img src="https://github.com/Jingyi-Wu-Richael/individualproject/blob/main/images/flow.png"  style="zoom:25%;" />

## Results
Our integrated models outperformed base versions in precision, recall, and F1 score metrics. The integration of Memory Modules in large language models highlights significant improvements in sentiment recognition. This methodology offers expansive applications in NLP, emphasizing its role in advancing sentiment recognition and related tasks.

## How to Run

1. Install the required packages. 
   ```
   pip install -r requirements.txt    
   ```
2. Ensure you have the emotion-emotion_69k.csv dataset in your working directory.
3. Run the Python script.

## Acknowledgements
I am deeply grateful to:
**Dr. Anandha Gopalan**, my esteemed advisor, for his unwavering support, mentoring, and invaluable intellectual insights which played a pivotal role in guiding this research.
**Teaching Assistant, Ruoyu Hu**, for his vital assistance, particularly in addressing technical challenges.
**Dr. Josiah Wang**, for his constructive feedback and enthusiasm that inspired a deeper exploration of the subject.
**The team at Facebook AI Research (FAIR)**, especially those involved in the EmpatheticDialogues dataset. Their dedication to open research and commitment to collaboration have enriched this project immeasurably.

## Reference
[1] Zhang Y, Sun S, Galley M, Chen YC, Brockett C, Gao X, et al. DIALOGPT: Large-scale generative pre-training for conversational response generation. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations; 2020. p. 270-8.

[2] Zhang S, Roller S, Goyal N, Artetxe M, Chen M, Chen S, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:220501068. 2022.

[3] Rashkin H, Smith EM, Li M, Boureau YL. Towards empathetic open-domain conversation models: A new benchmark and dataset. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics; 2018. p. 5370-81.
