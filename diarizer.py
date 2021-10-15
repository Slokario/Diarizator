
# import subprocess
# from sklearn.cluster import AgglomerativeClustering

import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torchaudio

from speechbrain.pretrained import EncoderClassifier
from tqdm.autonotebook import tqdm

from .cluster import cluster_AHC, cluster_SC
from .utils import (check_wav_16khz_mono, convert_wavfile,
                     parse_ttml)


class Diarizer:

    def __init__(self,
                 embed_model='xvec',
                 cluster_method='sc',
                 window=1.5,
                 period=0.75):

        assert embed_model in [
            'xvec', 'ecapa'], "Можно использовать только xvec или ecapa для embeding"
        assert cluster_method in [
            'ahc', 'sc'], "Можно использовать только ahc или sc для кластеризации"

        if cluster_method == 'ahc':
            self.cluster = cluster_AHC
        if cluster_method == 'sc':
            self.cluster = cluster_SC

        self.vad_model, self.get_speech_ts = self.setup_VAD()

        self.run_opts = {"device": "cuda:0"} if torch.cuda.is_available() else {
            "device": "cpu"}

        if embed_model == 'xvec':
            self.embed_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                              savedir="pretrained_models/spkrec-xvect-voxceleb",
                                                              run_opts=self.run_opts)
        if embed_model == 'ecapa':
            self.embed_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                              savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                                              run_opts=self.run_opts)

        self.window = window
        self.period = period

    def setup_VAD(self):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad')
        # force_reload=True)

        (get_speech_ts,
            _, _, read_audio,
            _, _, _) = utils
        return model, get_speech_ts

    def vad(self, signal):
        """
        Runs the VAD model on the signal
        """
        return self.get_speech_ts(signal, self.vad_model)

    def windowed_embeds(self, signal, fs, window=1.5, period=0.75):
        """
        Вычисляет embeddings окон сигнала

        window: размер окна в секундах
        period: перекрытие в секундах

        returns: embeddings, segment info
        """
        len_window = int(window * fs)
        len_period = int(period * fs)
        len_signal = signal.shape[1]

        # Get the windowed segments
        segments = []
        start = 0
        while start + len_window < len_signal:
            segments.append([start, start+len_window])
            start += len_period

        segments.append([start, len_signal-1])
        embeds = []

        with torch.no_grad():
            for i, j in segments:
                signal_seg = signal[:, i:j]
                seg_embed = self.embed_model.encode_batch(signal_seg)
                embeds.append(seg_embed.squeeze(0).squeeze(0).cpu().numpy())

        embeds = np.array(embeds)
        return embeds, np.array(segments)

    def recording_embeds(self, signal, fs, speech_ts):
        """
        Принимает сигнал и вывод VAD (speech_ts) и производит оконные вложения

        returns: embeddings, segment info
        """
        all_embeds = []
        all_segments = []

        for utt in tqdm(speech_ts, desc='Utterances', position=0):
            start = utt['start']
            end = utt['end']

            utt_signal = signal[:, start:end]
            utt_embeds, utt_segments = self.windowed_embeds(utt_signal,
                                                            fs,
                                                            self.window,
                                                            self.period)
            all_embeds.append(utt_embeds)
            all_segments.append(utt_segments + start)

        all_embeds = np.concatenate(all_embeds, axis=0)
        all_segments = np.concatenate(all_segments, axis=0)
        return all_embeds, all_segments

    @staticmethod
    def join_segments(cluster_labels, segments, tolerance=5):
        """
        Использует среднюю точку для разрешения конфликтов перекрытия,
        позволяет комбинировать очень минимально разделенные сегменты ( в samples)
        """
        assert len(cluster_labels) == len(segments)

        new_segments = [{'start': segments[0][0],
                         'end': segments[0][1],
                         'label': cluster_labels[0]}]

        for l, seg in zip(cluster_labels[1:], segments[1:]):
            start = seg[0]
            end = seg[1]

            protoseg = {'start': seg[0],
                        'end': seg[1],
                        'label': l}

            if start <= new_segments[-1]['end']:
                # If segments overlap
                if l == new_segments[-1]['label']:
                    # If overlapping segment has same label
                    new_segments[-1]['end'] = end
                else:
                    # If overlapping segment has diff label
                    # Resolve by setting new start to midpoint
                    # And setting last segment end to midpoint
                    overlap = new_segments[-1]['end'] - start
                    midpoint = start + overlap//2
                    new_segments[-1]['end'] = midpoint
                    protoseg['start'] = midpoint
                    new_segments.append(protoseg)
            else:
                # If there's no overlap just append
                new_segments.append(protoseg)

        return new_segments

    @staticmethod
    def make_output_seconds(cleaned_segments, fs):
        """
        Преобразование очищенных сегментов в читаемый формат (в секундах)
        """
        for seg in cleaned_segments:
            seg['start_sample'] = seg['start']
            seg['end_sample'] = seg['end']
            seg['start'] = seg['start']/fs
            seg['end'] = seg['end']/fs
        return cleaned_segments

    def diarize(self,
                wav_file,
                num_speakers=2,
                threshold=None,
                silence_tolerance=0.2,
                enhance_sim=True,
                extra_info=False):
        """
        Диаризация mono-wav 16кГц файла, создает список сегментов

            Inputs:
                wav_file (path): файл или путь к файлу
                num_speakers (int) или NoneType: количество спикеров
                threshold (float) или NoneType: пороговое значение для кластеризации,
                                                если неизвестно количество говорящих
                silence_tolerance (float): бъединяет сегменты в один, если тишина между ними меньше silence_tolerance
                enhance_sim (bool): улучшение вычисления матрици аффинности для спектральной кластеризации
                extra_info (bool): Возвращает дополнительные данные при диаризации, которые не используются в cleaned_segments


            Outputs:
                If extra_info is False:
                    segments (list): список с информации о сегментах вида:
                              {
                                'start': Start time of segment in seconds,
                                'start_sample': Starting index of segment,
                                'end': End time of segment in seconds,
                                'end_sample' Ending index of segment,
                                'label': Cluster label of segment
                              }

        """
        recname = os.path.splitext(os.path.basename(wav_file))[0] # если есть путь c:\papka1\papka2\textT.txt , то recname = textT

        if check_wav_16khz_mono(wav_file):
             signal, fs = torchaudio.load(wav_file)
        # else:
        #      print("Converting audio file to single channel WAV using ffmpeg...")
        #      converted_wavfile = os.path.join(os.path.dirname(
        #          wav_file), '{}_converted.wav'.format(recname))
        #      convert_wavfile(wav_file, converted_wavfile)
        #      assert os.path.isfile(
        #          converted_wavfile), "Couldn't find converted wav file, failed for some reason"
        #      signal, fs = torchaudio.load(converted_wavfile)

        print('Running VAD...')
        speech_ts = self.vad(signal[0])
        print('Splitting by silence found {} utterances'.format(len(speech_ts)))
        assert len(speech_ts) >= 1, "Couldn't find any speech during VAD"

        print('Extracting embeddings...')
        embeds, segments = self.recording_embeds(signal, fs, speech_ts)

        print('Clustering to {} speakers...'.format(num_speakers))
        cluster_labels = self.cluster(embeds, n_clusters=num_speakers,
                                      threshold=threshold, enhance_sim=enhance_sim)

        print('Cleaning up output...')
        cleaned_segments = self.join_segments(cluster_labels, segments)
        cleaned_segments = self.make_output_seconds(cleaned_segments, fs)
        cleaned_segments = self.join_samespeaker_segments(cleaned_segments,
                                                          silence_tolerance=silence_tolerance)
        print('Готово!')

        if not extra_info:
            return cleaned_segments
        else:
            return cleaned_segments, embeds, segments


    @staticmethod
    def join_samespeaker_segments(segments, silence_tolerance=0.5):
        """
        Объедините сегменты, принадлежащие одному говорящему
        Если тишина больше silence_tolerance, то не присоединяет
        """
        new_segments = [segments[0]]

        for seg in segments[1:]:
            if seg['label'] == new_segments[-1]['label']:
                if new_segments[-1]['end'] + silence_tolerance >= seg['start']:
                    new_segments[-1]['end'] = seg['end']
                    new_segments[-1]['end_sample'] = seg['end_sample']
                else:
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
        return new_segments


