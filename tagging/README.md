## Table of Contents

- [Definition of hard samples](#definition-of-hard-samples)
- [Definition of tags](#definition-of-tags)
- [Annotated tags](#annotated-tags)

## Definition of hard samples

We define the “hard samples” in the non-blind and blind test datasets as those being poorly enhanced by **at least two** participating teams.
A poorly enhanced sample is defined using the algorithm below:
1. For each degraded speech sample, we gather all the evaluation metrics of the corresponding enhanced speech from each participating team.
  > ```python
  > metrics = {
  >     "DNSMOS": {
  >         "Team1": [3.12, 2.24, 2.07, ...],  # length: num_samples
  >         "Team2": [3.08, 2.51, 2.13, ...],  # length: num_samples
  >         ...
  >     },
  >     "PESQ": {
  >         "Team1": [2.55, 2.06, 1.84, ...],  # length: num_samples
  >         "Team2": [2.43, 2.19, 1.92, ...],  # length: num_samples
  >         ...
  >     },
  >     "ESTOI": {
  >         "Team1": [0.85, 0.80, 0.78, ...],  # length: num_samples
  >         "Team2": [0.84, 0.82, 0.79, ...],  # length: num_samples
  >         ...
  >     },
  >     ...
  > }
  > ```
2. For each metric, we correspondingly predefine a threshold and a weight. We then check whether the metric value of the enhanced speech is above the threshold (only for MCD and LSD) / below the threshold (for other metrics), which indicates that the enhanced speech may be low-quality.
  > The thresholds for each metric are as follows:
  > ```python
  > # for both blind and non-blind test sets
  > METRIC_THRESHOLDS = {
  >     "DNSMOS": 2.0,             # < this ⇒ low-quality
  >     "NISQA": 2.0,              # < this ⇒ low-quality
  >     "PESQ": 1.5,               # < this ⇒ low-quality
  >     "ESTOI": 0.6,              # < this ⇒ low-quality
  >     "SDR": 0.0,                # < this ⇒ low-quality
  >     "MCD": 5.0,   # > this ⇒ low-quality
  >     "LSD": 5.0,   # > this ⇒ low-quality
  >     "POLQA": 1.7,              # < this ⇒ low-quality
  >     "SpeechBERTScore": 0.5,    # < this ⇒ low-quality
  >     "LPS": 0.4,                # < this ⇒ low-quality
  >     "SpkSim": 0.4,             # < this ⇒ low-quality
  >     "WAcc": 0.5,               # < this ⇒ low-quality
  >     "MOS": 2.0,                # < this ⇒ low-quality
  > }
  >
  > # only for non-blind test set
  > METRIC_WEIGHTS = {
  >     # (1) Non-intrusive SE metrics
  >     "DNSMOS": 1/8,
  >     "NISQA": 1/8,
  >     # (2) Intrusive SE metrics
  >     "PESQ": 1/20,
  >     "ESTOI": 1/20,
  >     "SDR": 1/20,
  >     "MCD": 1/20,
  >     "LSD": 1/20,
  >     # (3) Downstream-task-independent metrics
  >     "SpeechBERTScore": 1/8,
  >     "LPS": 1/8,
  >     # (4) Downstream-task-dependent metrics
  >     "SpkSim": 1/8,
  >     "WAcc": 1/8,
  > }
  >
  > # only for blind test set
  > METRIC_WEIGHTS = {
  >     # (1) Non-intrusive SE metrics
  >     "DNSMOS": 1/10,
  >     "NISQA": 1/10,
  >     # (2) Intrusive SE metrics
  >     "PESQ": 1/30,
  >     "ESTOI": 1/30,
  >     "SDR": 1/30,
  >     "MCD": 1/30,
  >     "LSD": 1/30,
  >     "POLQA": 1/30,
  >     # (3) Downstream-task-independent metrics
  >     "SpeechBERTScore": 1/10,
  >     "LPS": 1/10,
  >     # (4) Downstream-task-dependent metrics
  >     "SpkSim": 1/10,
  >     "WAcc": 1/10,
  >     # (5) Subjective SE metrics
  >     "MOS": 1/5,
  > }
  >
  > def is_low_quality(metric_name, metric_value):
  >     threshold = METRIC_THRESHOLDS[metric_name]
  >     if metric_name in ["MCD", "LSD"]:
  >         return metric_value > threshold
  >     else:
  >         return metric_value < threshold
  > ```
3. For each participating team on each utterance: if a metric value indicates low quality, we record it as a negative vote for the current metric with a corresponding weight; otherwise we record it as a positive vote with a corresponding weight.<br/>The overall quality of the utterance is then calculated as the sum of the votes from all metrics.<br/>If the overall quality is low (i.e., negative), we consider the utterance as a hard sample for that team.
  > ```python
  > uid_counter = {}
  > for team in ALL_TEAM_IDS:  # iterate through all participating teams
  >     for u, uid in enumerate(ALL_UIDS):  # iterate through all utterances
  >         vote = []  # list to store quality votes for each metric
  >         for name in ALL_METRICS:  # iterate through all metrics
  >             if math.isnan(metrics[name][team][u]):
  >                 vote.append(0)
  >             elif is_low_quality(name, metrics[name][team][u]):
  >                 vote.append(-METRIC_WEIGHTS[name]) # negative weight for low-quality
  >             else:
  >                 vote.append(METRIC_WEIGHTS[name])  # positive weight for not-too-low-quality
  >         # overall quality = weights sum of per-metric quality votes
  >         if sum(vote) < 0:
  >             # overall quality < 0 ⇒ low-quality utterance
  >             uidCounter.setdefault(uid, set()).add(team)
  > ```

4. Finally, for each utterance, we check whether it has been marked as low-quality by at least two teams. If so, we add it to the list of global hard samples.
  > ```python
  > hard_samples = set()
  > for uid, teams in uidCounter.items():
  >     if len(teams) >= 2:
  >         # this utterance is marked as low-quality by at least two teams
  >         hard_samples.add(uid)
  > ```

The final lists of hard samples on the non-blind and blind test sets are provided below.

<details><summary>1. Hard samples (#=93) detected in non-blind test set</summary><div>

<table>
<td>
fileid_137, fileid_15, fileid_16, fileid_204, fileid_229, fileid_240, fileid_280, fileid_300, fileid_309, fileid_338, fileid_394, fileid_413, fileid_433, fileid_451, fileid_460, fileid_464, fileid_466, fileid_467, fileid_475, fileid_480, fileid_481, fileid_483, fileid_497, fileid_499, fileid_502, fileid_504, fileid_508, fileid_510, fileid_511, fileid_516, fileid_518, fileid_537, fileid_551, fileid_559, fileid_564, fileid_568, fileid_574, fileid_576, fileid_640, fileid_641, fileid_658, fileid_666, fileid_671, fileid_672, fileid_675, fileid_682, fileid_684, fileid_687, fileid_704, fileid_707, fileid_710, fileid_719, fileid_722, fileid_724, fileid_727, fileid_729, fileid_73, fileid_753, fileid_754, fileid_759, fileid_765, fileid_768, fileid_77, fileid_771, fileid_773, fileid_774, fileid_797, fileid_801, fileid_802, fileid_808, fileid_811, fileid_814, fileid_816, fileid_820, fileid_830, fileid_833, fileid_835, fileid_837, fileid_839, fileid_841, fileid_843, fileid_844, fileid_845, fileid_847, fileid_849, fileid_850, fileid_852, fileid_857, fileid_858, fileid_859, fileid_862, fileid_926, fileid_947
</td>
</table>

</div></details>


<details><summary>2. Hard samples (#=328) detected in blind test set</summary><div>

<table>
<td>
fileid_1, fileid_100, fileid_106, fileid_109, fileid_110, fileid_111, fileid_114, fileid_116, fileid_119, fileid_12, fileid_120, fileid_123, fileid_130, fileid_135, fileid_139, fileid_141, fileid_142, fileid_143, fileid_155, fileid_156, fileid_157, fileid_161, fileid_162, fileid_163, fileid_171, fileid_172, fileid_173, fileid_177, fileid_179, fileid_182, fileid_183, fileid_186, fileid_190, fileid_203, fileid_204, fileid_209, fileid_211, fileid_212, fileid_216, fileid_22, fileid_222, fileid_226, fileid_228, fileid_229, fileid_23, fileid_231, fileid_242, fileid_245, fileid_250, fileid_252, fileid_254, fileid_258, fileid_260, fileid_261, fileid_262, fileid_266, fileid_268, fileid_269, fileid_270, fileid_273, fileid_274, fileid_277, fileid_278, fileid_279, fileid_28, fileid_282, fileid_297, fileid_298, fileid_300, fileid_301, fileid_313, fileid_317, fileid_318, fileid_322, fileid_325, fileid_33, fileid_331, fileid_337, fileid_341, fileid_344, fileid_346, fileid_35, fileid_351, fileid_352, fileid_354, fileid_364, fileid_367, fileid_37, fileid_374, fileid_379, fileid_387, fileid_388, fileid_389, fileid_391, fileid_394, fileid_395, fileid_397, fileid_4, fileid_40, fileid_400, fileid_401, fileid_402, fileid_404, fileid_408, fileid_413, fileid_414, fileid_416, fileid_421, fileid_424, fileid_427, fileid_428, fileid_429, fileid_43, fileid_435, fileid_44, fileid_440, fileid_444, fileid_445, fileid_449, fileid_450, fileid_456, fileid_457, fileid_459, fileid_46, fileid_471, fileid_480, fileid_49, fileid_492, fileid_499, fileid_5, fileid_50, fileid_501, fileid_508, fileid_51, fileid_510, fileid_511, fileid_512, fileid_516, fileid_518, fileid_519, fileid_520, fileid_525, fileid_53, fileid_530, fileid_532, fileid_533, fileid_534, fileid_535, fileid_54, fileid_541, fileid_543, fileid_546, fileid_548, fileid_552, fileid_557, fileid_565, fileid_566, fileid_567, fileid_573, fileid_586, fileid_587, fileid_589, fileid_59, fileid_590, fileid_596, fileid_598, fileid_599, fileid_603, fileid_605, fileid_606, fileid_607, fileid_608, fileid_610, fileid_615, fileid_616, fileid_62, fileid_622, fileid_624, fileid_627, fileid_628, fileid_629, fileid_63, fileid_630, fileid_631, fileid_632, fileid_634, fileid_636, fileid_637, fileid_640, fileid_645, fileid_648, fileid_649, fileid_65, fileid_653, fileid_654, fileid_657, fileid_659, fileid_667, fileid_668, fileid_671, fileid_674, fileid_675, fileid_679, fileid_682, fileid_686, fileid_691, fileid_692, fileid_695, fileid_697, fileid_699, fileid_7, fileid_70, fileid_704, fileid_706, fileid_707, fileid_708, fileid_710, fileid_711, fileid_715, fileid_717, fileid_718, fileid_719, fileid_72, fileid_722, fileid_723, fileid_728, fileid_729, fileid_732, fileid_74, fileid_740, fileid_751, fileid_752, fileid_755, fileid_756, fileid_757, fileid_758, fileid_759, fileid_762, fileid_77, fileid_771, fileid_772, fileid_774, fileid_777, fileid_780, fileid_781, fileid_782, fileid_785, fileid_786, fileid_787, fileid_788, fileid_790, fileid_791, fileid_794, fileid_796, fileid_798, fileid_799, fileid_800, fileid_801, fileid_802, fileid_804, fileid_806, fileid_809, fileid_810, fileid_815, fileid_816, fileid_818, fileid_820, fileid_825, fileid_834, fileid_84, fileid_841, fileid_843, fileid_847, fileid_848, fileid_854, fileid_855, fileid_857, fileid_865, fileid_867, fileid_869, fileid_872, fileid_874, fileid_878, fileid_88, fileid_880, fileid_882, fileid_888, fileid_889, fileid_89, fileid_898, fileid_899, fileid_907, fileid_91, fileid_914, fileid_917, fileid_918, fileid_919, fileid_920, fileid_925, fileid_926, fileid_927, fileid_932, fileid_936, fileid_937, fileid_939, fileid_942, fileid_946, fileid_949, fileid_95, fileid_951, fileid_953, fileid_954, fileid_959, fileid_963, fileid_964, fileid_970, fileid_972, fileid_973, fileid_974, fileid_976, fileid_979, fileid_980, fileid_983, fileid_984, fileid_985, fileid_987, fileid_989, fileid_996
</td>
</table>

</div></details>

## Definition of tags

Below defines the tags used in the hard sample tagging process. The tags are divided into several categories, and each category contains multiple tags. Except for the "SNR" category, all tags were manually annotated via informal listening. The "SNR" category is automatically annotated using the meta information of the datasets (only for simulated samples).

<table>
<thead>
<tr>
    <th>Category</th>
    <th>Tag</th>
    <th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
    <td rowspan="2">Type</td>
    <td>simulated</td>
    <td>The degraded speech sample is simulated.</td>
</tr>
<tr>
    <td>real_recording</td>
    <td>The degraded speech sample is a real recording.</td>
</tr>
<tr>
    <td rowspan="3">Intelligibility</td>
    <td>high_intelligibility</td>
    <td>The speech can be heard clearly and understood easily (even in the presence of noise/interference).</td>
</tr>
<tr>
    <td>medium_intelligibility</td>
    <td>The speech can be heard and understood by listening carefully.</td>
</tr>
<tr>
    <td>low_intelligibility</td>
    <td>The speech speech cannot be understood.</td>
</tr>
<tr>
    <td rowspan="3">Voice style</td>
    <td>reading_style</td>
    <td>The audio contains reading-style voice.</td>
</tr>
<tr>
    <td>spontaneous_style</td>
    <td>The audio contains spontaneous (conversational) speech.</td>
</tr>
<tr>
    <td>singing_voice</td>
    <td>The audio contains singing voice.</td>
</tr>
<tr>
    <td rowspan="5">Speaker-related</td>
    <td>male</td>
    <td>The voice comes from a male speaker.</td>
</tr>
<tr>
    <td>female</td>
    <td>The voice comes from a female speaker.</td>
</tr>
<tr>
    <td>Nspk</td>
    <td>At least 2 different voices appear sequentially in the audio.</td>
</tr>
<tr>
    <td>speech_overlap</td>
    <td>There are overlapped voices in the audio.</td>
</tr>
<tr>
    <td>unison</td>
    <td>There is unison in the audio.</td>
</tr>
<tr>
    <td rowspan="4">Specific degradations</td>
    <td>bandwidth_limitation</td>
    <td>Bandwidth limitation can be heard / seen from the spectrogram.</td>
</tr>
<tr>
    <td>clipping</td>
    <td>Clipping can be heard / seen from the waveform.</td>
</tr>
<tr>
    <td>sound_effect</td>
    <td>Sound effects can be heard.</td>
</tr>
<tr>
    <td>music</td>
    <td>Background music can be heard.</td>
</tr>
<tr>
    <td rowspan="4">Noise type</td>
    <td>low_freq_noise</td>
    <td>Low-frequency narrow-band noise can be heard / seen from the spectrogram.</td>
</tr>
<tr>
    <td>high_freq_noise</td>
    <td>High-frequency narrow-band noise can be heard / seen from the spectrogram.</td>
</tr>
<tr>
    <td>background_noise</td>
    <td>Stationary wide-band background noise can be heard / seen from the spectrogram.</td>
</tr>
<tr>
    <td>instantaneous_noise</td>
    <td>Instantaneous noise can be heard / seen from the spectrogram.</td>
</tr>
<tr>
    <td rowspan="4">SNR</td>
    <td>SNR (-5~0 dB)</td>
    <td>The SNR of the simulated sample is between -5 dB and 0 dB.</td>
</tr>
<tr>
    <td>SNR (0~5 dB)</td>
    <td>The SNR of the simulated sample is between 0 dB and 5 dB.</td>
</tr>
<tr>
    <td>SNR (5~10 dB)</td>
    <td>The SNR of the simulated sample is between 5 dB and 10 dB.</td>
</tr>
<tr>
    <td>SNR (>10 dB)</td>
    <td>The SNR of the simulated sample is higher than 10 dB.</td>
</tr>
<tr>
    <td rowspan="3">Reverberation</td>
    <td>weak_reverberation</td>
    <td>Weak reverberation exists in the audio.</td>
</tr>
<tr>
    <td>medium_reverberation</td>
    <td>Medium-level reverberation is perceptible in the audio.</td>
</tr>
<tr>
    <td>strong_reverberation</td>
    <td>Strong reverberation is perceptible in the audio.</td>
</tr>
</tbody>
</table>


## Annotated tags

The annotation of tags was performed by the author of this repository. The tags for the original degraded speech in the non-blind and blind test sets are provided in [nonblind_test_tags.tsv](nonblind_test_tags.tsv) and [blind_test_tags.tsv](blind_test_tags.tsv), respectively.

The files are tab-separated values (TSV) files, and the first row contains the header.
The first column of the files contains the file IDs of the degraded speech samples, and the second column contains the tags.
The tags are separated by semi-colons `;`.