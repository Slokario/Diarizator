
# import os
from pprint import pprint
# import math
# from tqdm.autonotebook import tqdm
# import numpy as np


from diarizer import Diarizer
from utils import (check_wav_16khz_mono, convert_wavfile, parse_ttml,
                                   waveplot, combined_waveplot, waveplot_perspeaker)

import matplotlib.pyplot as plt

import soundfile as sf

from IPython.display import Audio, display



NUM_SPEAKERS = 2
WAV_FILE = 'Dialog.wav'
signal, fs = sf.read(WAV_FILE)
waveplot(signal, fs, figsize=(20, 3))
#plt.show()

diar = Diarizer(
                embed_model='xvec', # supported types: ['xvec', 'ecapa']
                cluster_method='ahc', # supported types: ['ahc', 'sc']
                window=1.5, # size of window to extract embeddings (in seconds)
                period=0.75 # hop of window (in seconds)
                )
segments = diar.diarize(WAV_FILE,
                        num_speakers=NUM_SPEAKERS)
combined_waveplot(signal, fs, segments, figsize=(10,3), tick_interval=60)
plt.show()
display(Audio(signal, rate=fs))
#waveplot_perspeaker(signal, fs, segments)
pprint(segments)