from faster_whisper import WhisperModel
import pyaudio
import wave
import os
import torch

def record_audio_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk_to_text(model, file_path):
    segments, _ = model.transcribe(file_path)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

def main():
    model_size = "small"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_size, device=device, compute_type="int8")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )
    
    print("Recording started. Press Ctrl+C to stop.")
    accumulated_transcription = ""

    try: 
        while True:
            chunk_file = "temp_chunk.wav"
            record_audio_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk_to_text(model, chunk_file)
            if transcription:
                print("Transcription:", transcription)
                accumulated_transcription += transcription + " "
            os.remove(chunk_file)

    except KeyboardInterrupt:
        print("\nTranscription ended.")
    finally:
        if accumulated_transcription:
            with open("log.txt", "w") as log_file:
                log_file.write(accumulated_transcription.strip())
            print("LOGGED TRANSCRIPTION: " + accumulated_transcription.strip())
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
