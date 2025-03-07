import whisper
import nltk
import spacy
from collections import defaultdict
import os
import json

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

def transcribe_audio(audio_path):
    """Convert speech to text using Whisper."""
    model = whisper.load_model("base")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    """Generate a summary of the meeting transcript."""
    sentences = nltk.tokenize.sent_tokenize(text)
    summary = " ".join(sentences[:max(1, len(sentences)//3)]) 
    return summary

def extract_tasks(text):
    """Extract tasks and assignments from the transcript."""
    tasks = defaultdict(list)
    doc = nlp(text)
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in ["task", "action", "deadline", "assign", "responsibility"]):
            entities = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
            task_description = sent.text.strip()
            if not entities:
                entities.append("Unassigned")
            for person in entities:
                tasks[person].append(task_description)
    return dict(tasks)

def save_summary(summary, file_path="summary.txt"):
    """Save the meeting summary to a text file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary)

def save_tasks(tasks, file_path="tasks.json"):
    """Save the extracted tasks to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4)

def load_previous_tasks(file_path="tasks.json"):
    """Load previous task assignments from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def merge_tasks(existing_tasks, new_tasks):
    """Merge previous and new tasks."""
    for person, task_list in new_tasks.items():
        existing_tasks.setdefault(person, []).extend(task_list)
    return existing_tasks

def main(audio_file):
    """Main function to process meeting audio."""
    try:
        transcript = transcribe_audio(audio_file)
        summary = summarize_text(transcript)
        new_tasks = extract_tasks(transcript)
        
        existing_tasks = load_previous_tasks()
        all_tasks = merge_tasks(existing_tasks, new_tasks)
        
        print("Meeting Summary:")
        print(summary)
        print("\nTask Assignments:")
        for person, task_list in all_tasks.items():
            print(f"{person}: {', '.join(task_list)}")
        
        save_summary(summary)
        save_tasks(all_tasks)
        print("\nSummary and tasks saved successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    audio_path = "people-talking-289717.mp3" 
    main(audio_path)
