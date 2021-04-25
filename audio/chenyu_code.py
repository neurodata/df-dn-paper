import numpy as np
import IPython.display as ipd
from joblib import Parallel, delayed

from spoken_digit_functions import *

path_recordings = 'recordings/'

audio_data, x_spec_mini, y_number, y_speaker = load_spoken_digit(path_recordings)


num = 2999 # choose from 0 to 2999

print('This is number', y_number[num], 'spoken by speaker', y_speaker[num].upper(), '\nDuration:',
      audio_data[num].shape[0], 'samples in', audio_data[num].shape[0] / 8000, 'seconds')
ipd.Audio(audio_data[num], rate=8000)




num = 9 # choose from 0 to 9
display_spectrogram(x_spec_mini, y_number, y_speaker, num)


x = x_spec_mini.reshape(3000,32,32,1)    # (3000,28,28,1)
y = y_number                             # (3000,)
y_speaker = y_speaker                    # (3000,), dtype: string


