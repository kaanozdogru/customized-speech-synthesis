# Customized Speech Synthesis with Latent Vector Manipulation


## Setup
Setup using pip:\
Python 3.7 is recommended
```
pip install -r requirements.txt
```


## Usage
### Pre-Trained models
Make sure the pre-trained models can be found at:
```
saved_models/default/encoder.pt
saved_models/default/synthesizer.pt
saved_models/default/vocoder.pt
```

### Transformation

To convert a sample noise run the following:

```

python transform_speaker.py --target_audio_dir speech_data/female \
                       --current_audio_dir speech_data/male \
                       --test_audio_path lebron.mp3
```
This will take the mp3 file of LeBron James speaking and convert his voice to a female voice. It will be saved as `transformed.wav`. If you want to try other transformations, create a folder under speech_data/ and populate it with around 20 recordings in a given class. Then replace the arguments above to run with the corresponding folders.

If you want to only do voice clonning, run the following, you can put any other path of a english speaking recording. The output file will be saved as `voice_clone.wav`.

```
python voice_clone.py --audio_path lebron.mp3
```