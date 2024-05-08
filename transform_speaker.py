import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--target_audio_dir", type=Path, default=Path("speech_data/target"),
                        help="Directory containing target audio files for processing.")
    parser.add_argument("--current_audio_dir", type=Path, default=Path("speech_data/current"),
                        help="Directory containing current audio files for processing.")
    parser.add_argument("--test_audio_path", type=Path, default=Path("test_audio.mp3"),
                        help="Path to a test audio file for voice manipulation.")

    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/default/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/default/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)


    ## Run a test
    print("Testing your configuration with small inputs.")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
    # returns, but here we're going to make one ourselves just for the sake of showing that it's
    # possible.
    embed = np.random.rand(speaker_embedding_size)
    # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
    # embeddings it will be).
    embed /= np.linalg.norm(embed)
    # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
    # illustrate that
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ["test 1", "test 2"]
    print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    mels = synthesizer.synthesize_spectrograms(texts, embeds)

    # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
    # can concatenate the mel spectrograms to a single one.
    mel = np.concatenate(mels, axis=1)
    # The vocoder can take a callback function to display the generation. More on that later. For
    # now we'll simply hide it like this:
    no_action = lambda *args: None
    print("\tTesting the vocoder...")
    # For the sake of making this test short, we'll pass a short target length. The target length
    # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
    # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
    # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
    # that has a detrimental effect on the quality of the audio. The default parameters are
    # recommended in general.
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

    print("All test passed! You can now synthesize speech.\n\n")


    print("Interactive generation loop")
    num_generated = 0

    def embed_mp3(path):
        # The following two methods are equivalent:
        # - Directly load from the filepath:
        preprocessed_wav = encoder.preprocess_wav(path)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(str(path))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")

        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embedding = encoder.embed_utterance(preprocessed_wav)
        return embedding
    
    def embed_audio_files(directory):
        embeddings = []
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]
        for file in files:
            print(f"Processing {file}...")
            preprocessed_wav = encoder.preprocess_wav(file)
            original_wav, sampling_rate = librosa.load(file, sr=None)
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            embedding = encoder.embed_utterance(preprocessed_wav)
            embeddings.append(embedding)
        return embeddings, files
    
    
    def plot_embeddings(embeddings, labels, title='Embeddings', filename='embeddings_plot.png'):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(set(labels)):
            idx = [i for i, val in enumerate(labels) if val == label]
            plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label)
        plt.title(title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)  # Save the plot as a PNG file
        plt.close()  #

    def plot_embeddings_label(embeddings, labels, files, title='Embeddings', filename='embeddings_plot.png'):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap("tab10", len(set(labels)))

        for i, label in enumerate(set(labels)):
            idx = [index for index, val in enumerate(labels) if val == label]
            plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, c=[colors(i)]*len(idx))
            for j in idx:
                plt.annotate(Path(files[j]).stem,
                            (reduced[j, 0], reduced[j, 1]),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center',
                            fontsize=8)

        plt.title(title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def train_svm(X, y):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train the SVM
        svm = SVC(kernel='linear')
        svm.fit(X_train, y_train)
        
        # Predictions and evaluations
        y_pred = svm.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return svm, scaler  # Return the trained model and the scaler

    def compute_gender_direction(female_embeddings, male_embeddings):
        # Compute the mean embedding vector for female and male samples
        mean_female = np.mean(female_embeddings, axis=0)
        mean_male = np.mean(male_embeddings, axis=0)
        
        # The direction vector from mean female to mean male
        direction = mean_male - mean_female
        direction /= np.linalg.norm(direction)  # Normalize the direction vector
        return direction

    def manipulate_embedding(embedding, direction, alpha):
        # Manipulate the embedding
        z_edit = embedding + alpha * direction
        z_edit /= np.linalg.norm(z_edit)  # Normalize the manipulated embedding
        return z_edit


    # try:
    # Get the reference audio filepath
    message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                "wav, m4a, flac, ...):\n"


    target_embeddings, target_files = embed_audio_files(str(args.target_audio_dir))
    curr_embeddings, curr_files = embed_audio_files(str(args.current_audio_dir))

    latent_direction = compute_gender_direction(target_embeddings, curr_embeddings)

    # Combine and label data
    
    # Example manipulation: Adjust a specific embedding to increase 'femaleness'
    test_embedding_path = str(args.test_audio_path)
    test_embedding = embed_mp3(test_embedding_path)
    manipulated_embedding = manipulate_embedding(test_embedding, latent_direction, alpha=-1.0)  # Increase femaleness
    embed = manipulated_embedding
    # Train the SVM
    # all_embeddings = np.vstack((female_embeddings, male_embeddings))
    # labels = ['Female']*len(female_embeddings) + ['Male']*len(male_embeddings)
    # svm_model, scaler = train_svm(all_embeddings, labels)

    # Plotting
    # plot_embeddings(all_embeddings, labels, title='Gender Voice Embeddings')


    # except Exception as e:
    #     print("Caught exception: %s" % repr(e))
    #     print("Restarting\n")
    ## Generating the spectrogram
    # text = input("Write a sentence (+-20 words) to be synthesized:\n")
    text = "Hello everyone, I hope everyone is having a good day. It is a great day in New York!"

    # If seed is specified, reset torch seed and force synthesizer reload
    if args.seed is not None:
        torch.manual_seed(args.seed)
        synthesizer = Synthesizer(args.syn_model_fpath)

    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")


    ## Generating the waveform
    print("Synthesizing the waveform:")

    # If seed is specified, reset torch seed and reload vocoder
    if args.seed is not None:
        torch.manual_seed(args.seed)
        vocoder.load_model(args.voc_model_fpath)

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav = vocoder.infer_waveform(spec)


    ## Post-generation
    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)

    # Play the audio (non-blocking)
    if not args.no_sound:
        import sounddevice as sd
        try:
            sd.stop()
            sd.play(generated_wav, synthesizer.sample_rate)
        except sd.PortAudioError as e:
            print("\nCaught exception: %s" % repr(e))
            print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
        except:
            raise

    # Save it on the disk
    filename = "transformed.wav" 
    print(generated_wav.dtype)
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    num_generated += 1
    print("\nSaved output as %s\n\n" % filename)


