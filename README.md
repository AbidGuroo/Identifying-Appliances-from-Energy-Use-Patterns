## Background
This plug load dataset contains current and voltage measurements sampled at 30 kHz from 11 different appliance types present in more than 60 households in Pittsburgh, Pennsylvania. Plug load refers to the energy used by products that are powered by means of an ordinary AC plug (i.e., plugged into an outlet). For each appliance, plug load measurements were post-processed to extract a two-second-long window of measurements of current and voltage. For some observations, the window contains both the startup transient state (turning the appliance on) as well as the steady-state operation (once the appliance is running). For others, the window only contains the steady-state operation. The observations were then transformed into two spectrograms, one for current, and one for voltage.

A spectrogram is a visual representation of the various frequencies of sound as they vary with time. The x-axis represents time (2 seconds in our case), and the y-axis represents frequency (measured in Hz). The colors indicate the amplitude of a particular frequency at a particular time (i.e., how loud it is). We're measuring amplitude in decibels, with 0 being the loudest, and -80 being the softest. So in the example spectrogram below, lower frequencies are louder than higher frequencies. Our spectrograms tend to have horizontal lines given that we are capturing appliances in their steady-state. In other words, the amplitudes of various frequencies are fairly constant over time.

### spectrograms
Spectrograms were created using librosa, a python package for music and audio analysis. The code to generate a spectrogram looks like this:

S = librosa.feature.melspectrogram(y=obs, sr=30000)
spectrogram = librosa.power_to_db(S)
plt.imsave(file_path, arr=spectrogram)
Under the hood, this process:

Takes the fourier transform of a windowed excerpt of the raw signal, in order to decompose the signal into its consistuent frequencies.
To learn more about fourier transforms, check out this awesome tutorial by 3Blue1Brown: But what is the Fourier Transform.
Maps the powers of the spectrum onto the mel scale. The mel scale is a perceptual scale where pitches are judged to be equal in distance from one another based on the human ear.
Takes the logs of the power (amplitude squared) at each of the mel frequencies to convert to decibel units.
Plots and saves the resulting image.
There is a lot of useful information encoded in these spectrograms. Now it's time to use your deep learning skills to parse out which patterns correspond to which types of appliances.
