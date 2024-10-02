# ZapAudio

On this library I explored some ML approaches to catalog my whatsapp audios to find them easily

I employed audio transcription with Whisper V3 in a GCloud virtual machine using the file [main.py](main.py)

After that I used some embedding features from BAAI/bge-m3 to compute similirities between the audios and find the best matches [calc_emebedding.py](calc_embedding.py).
