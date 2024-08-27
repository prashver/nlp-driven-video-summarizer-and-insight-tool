from flask import Flask, request, render_template, send_file
import os
from modules.setup import setup_nltk_resources, suppress_warnings
from modules.download_audio import download_audio_from_youtube
from modules.transcribe_audio import transcribe_audio
from modules.summarize_text import summarize_text
from modules.ner import perform_ner
from modules.keyword_extraction import extract_keywords
from modules.topic_modeling import preprocess_and_split, build_lda_model, print_and_save_topics

app = Flask(__name__)
app.secret_key = 'YOUR_SECRET_KEY'

# output directory
output_dir = "data/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def clear_old_results():
    for file_name in ["transcription.txt", "summary.txt", "entities.txt", "keywords.txt", "topics.txt", "podcast_audio.mp3"]:
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_podcast():
    setup_nltk_resources()
    suppress_warnings()

    youtube_url = request.form['youtube_url']

    if youtube_url:
        try:
            # Clear previous results
            clear_old_results()

            # Step 1: Download
            download_audio_from_youtube(youtube_url)
            audio = "data/output/podcast_audio.mp3"

            # Step 2: Transcribe
            transcription, detected_language = transcribe_audio(audio)
            with open(os.path.join(output_dir, "transcription.txt"), "w", encoding="utf-8") as file:
                file.write(f"Detected Language: {detected_language}\n\n")
                file.write(transcription)

            # Step 3: Summarize
            summary = summarize_text(transcription)
            with open(os.path.join(output_dir, "summary.txt"), "w") as file:
                file.write(summary)

            # Step 4: Perform NER
            entities = perform_ner(transcription)
            with open(os.path.join(output_dir, "entities.txt"), "w") as file:
                for entity in entities:
                    file.write(f"Entity: {entity[0]}, Label: {entity[1]}\n")

            # Step 5: Extract Keywords
            keywords = extract_keywords(transcription)
            with open(os.path.join(output_dir, "keywords.txt"), "w") as file:
                for keyword in keywords:
                    file.write(f"{keyword[0]}: {keyword[1]}\n")

            # Step 6: Perform Topic Modeling
            processed_text = preprocess_and_split(transcription)
            lda_topics = build_lda_model(processed_text)
            print_and_save_topics(lda_topics, num_words=3, output_file="topics.txt", output_dir="data/output")

            # Load results from files
            with open(os.path.join(output_dir, "transcription.txt"), "r", encoding="utf-8") as file:
                transcription = file.read()
            with open(os.path.join(output_dir, "summary.txt"), "r") as file:
                summary = file.read()
            with open(os.path.join(output_dir, "entities.txt"), "r") as file:
                entities = file.read()
            with open(os.path.join(output_dir, "keywords.txt"), "r") as file:
                keywords = file.read()
            with open(os.path.join(output_dir, "topics.txt"), "r") as file:
                topics = file.read()

            # Render the index page with results
            return render_template('index.html',
                                   audio_url='/get_audio',
                                   transcription=transcription,
                                   summary=summary,
                                   entities=entities,
                                   keywords=keywords,
                                   topics=topics,
                                   detected_language=detected_language,
                                   processing_complete=True)

        except Exception as e:
            # Handle errors
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html', error="Please enter a valid YouTube URL.")

# Route to serve the audio file
@app.route('/get_audio')
def get_audio():
    audio_path = os.path.join(output_dir, "podcast_audio.mp3")
    return send_file(audio_path, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
