from transformers import pipeline
from nltk.tokenize import word_tokenize, sent_tokenize

def summarize_text(transcription):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    target_length = 150
    max_chunk_length = 512
    chunks = [transcription[i:i + max_chunk_length] for i in range(0, len(transcription), max_chunk_length)]

    summaries = []
    total_words = 0
    for chunk in chunks:
        words_left = max(target_length - total_words, 30)
        if words_left <= 0:
            break

        input_length = len(word_tokenize(chunk))
        max_length = min(input_length - 1, words_left * 2)
        min_length = min(max_length // 2, words_left)

        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summary_text = summary[0]['summary_text']
        sentences = sent_tokenize(summary_text)

        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if total_words + len(sentence_words) > target_length:
                break
            summaries.append(sentence)
            total_words += len(sentence_words)

        if total_words >= target_length:
            break

    final_summary = " ".join(summaries)
    print(f"Summary ({len(word_tokenize(final_summary))} words):\n", final_summary)
    return final_summary
