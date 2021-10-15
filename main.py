
# import os
from pprint import pprint
# import math
# from tqdm.autonotebook import tqdm
# import numpy as np
#from IPython.display import Audio, display

from diarizer import Diarizer
from utils import (check_wav_16khz_mono, convert_wavfile, parse_ttml,
                                   waveplot, combined_waveplot, waveplot_perspeaker)

import matplotlib.pyplot as plt
import soundfile as sf




NUM_SPEAKERS = 2
WAV_FILE = 'Dialog.wav'
signal, fs = sf.read(WAV_FILE)
waveplot(signal, fs, figsize=(20, 3))
#plt.show()

diar = Diarizer(
                embed_model='xvec', # возможные варианты: ['xvec', 'ecapa']
                cluster_method='ahc', # возможные варианты: ['ahc', 'sc']
                window=1.5, # размер окна (в секундах)
                period=0.75 # перекрытие (в секундах)
                )
segments = diar.diarize(WAV_FILE,
                        num_speakers=NUM_SPEAKERS)

combined_waveplot(signal, fs, segments, figsize=(10,3), tick_interval=60)
plt.show()



#waveplot_perspeaker(signal, fs, segments)
#pprint(segments)