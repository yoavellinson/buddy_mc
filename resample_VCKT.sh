export SRC=/dsi/gannot-lab/gannot-lab1/datasets/VCTK/DS_10283_3443/VCTK-Corpus-0.92/wav48_silence_trimmed
export DST=/dsi/gannot-lab/gannot-lab1/datasets/VCTK/DS_10283_3443/VCTK-Corpus-0.92/wav16k

find "$SRC" -name "*mic2.flac" | \
parallel -j 12 '
    file={}
    rel=${file#'"$SRC"'/}
    out='"$DST"'/$rel
    out=${out%.flac}.wav
    mkdir -p "$(dirname "$out")"
    sox "$file" -r 16000 "$out"
'