import os 
import pandas as pd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datetime import datetime



if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
	return_timestamps=True
    )

    # Iterate over the audio folder

    files = os.listdir('files_original/')
    files = [f'files_original/{x}' for x in files]


    initial = []
    counter = 0
    for i in files:
        counter += 1
        try:
            init = datetime.now()

            result = pipe(i)
            transcribed = result["text"]

            initial.append({'file': i, 'text': transcribed})
            end = datetime.now()
            passed = end - init
            print(f'File: {counter} of {len(files)} processed time {passed.total_seconds()}')


        except:
            print(f'Error{i}')

    df = pd.DataFrame(initial)
    df.to_csv('transcribed_files.csv')
