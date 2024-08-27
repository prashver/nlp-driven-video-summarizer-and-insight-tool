# Podcast Video Summarizer And Insight Tool

## Overview

This project is an **NLP-driven** tool designed to automatically transcribe, summarize, and index podcasts, making it easier for users to find and understand content. Leveraging advanced Natural Language Processing techniques, the tool supports multilingual podcast transcription with translation to English. It also uses NLP models to extract key entities, identify important keywords, and analyze topics discussed. A Flask app is built for easy user interaction.

### Features

- **Video to Audio Conversion**: Converts podcast video to audio file and downloads them for processing.
- **Speech-to-Text**: Transcribes podcast audio of any language into English text using OpenAI's Whisper model.
- **Translation**: Translates non-English transcripts to English using a Hugging Face translation pipeline.
- **Text Summarization**: Generates concise summaries of podcast episode using Hugging Face's pre-trained NLP model.
- **Named Entity Recognition (NER)**: Utilizes spaCy's NLP library to extract key entities from transcript.
- **Keyword Extraction**: Employs NLP to identify important keywords or phrases to index and search episodes.
- **Topic Modeling**: Identifies the main topics discussed in a episode using LDA (Latent Dirichlet Allocation).
- **Flask App**: Provides a user-friendly interface for interacting with the NLP-powered tool.

### Project Execution

To run the application locally, follow these steps:
1. Clone this repository to your local machine : 
```
git clone https://github.com/prashver/nlp-driven-podcast-summarizer-and-insight-tool.git
cd nlp-driven-podcast-summarizer-and-insight-tool
```

3. Create a virtual environment:
```
python -m venv venv
venv\Scripts\activate
```

5. Install the required dependencies : 
```
pip install -r requirements.txt
```

7. Open and run app.py file : 
```
python app.py
```

### Usage

- Enter the copied video url in the url input field.
- Click on Process.
- Wait for some time and your desired results will get displayed on web page and, also get saved in directory.

https://github.com/user-attachments/assets/2811d3b4-1cdd-4219-a7cc-dcb46a32342e

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are welcome.

## License

This project is licensed under the [MIT License](LICENSE).

