#!/bin/bash

# 경로 설정
PT_DIR="/home/yjcho/EmpaTalk/bank_debug/train-code-dia-238000pt/"
VIDEO_DIR="/mnt/HDD_raid1/AvaMERG/video_v5_0"
AUDIO_DIR="/mnt/HDD_raid1/AvaMERG/audio_v5_0"
OUT_DIR="/home/yjcho/EmpaTalk/output/238000pt/train-code-dia-48"
REF_IMG="/home/yjcho/EmpaTalk/test_data/identity_source.jpg"

# 출력 디렉토리 생성
mkdir -p "$OUT_DIR"

# PT 파일 순회
for pt_file in "$PT_DIR"/bank_*.pt; do
    # 1. 파일명에서 dia... 정보 추출 (bank_와 .pt 제거)
    # 예: bank_dia08645utt1_16.pt -> dia08645utt1_16
    base_name=$(basename "$pt_file")
    dia_info=${base_name#bank_}  # 앞의 bank_ 제거
    dia_info=${dia_info%.pt}    # 뒤의 .pt 제거

    # 2. Listener 정보 설정
    LISTENER_V_PATH="$VIDEO_DIR/${dia_info}.mp4"
    LISTENER_A_PATH="$AUDIO_DIR/${dia_info}.wav"

    # 3. Speaker 정보 추론 (utt 번호 -1 찾기)
    # dia08645utt1_16 -> dia08645, 1, 16 분리
    prefix=$(echo "$dia_info" | grep -oP 'dia\d+')
    current_utt=$(echo "$dia_info" | grep -oP 'utt\d+' | sed 's/utt//')
    prev_utt=$((current_utt - 1))

    # Speaker 파일 찾기 (diaXXXXXutt{N-1}_*.mp4)
    # _ 뒤의 숫자가 무엇이든 상관없으므로 첫 번째 매칭되는 파일을 가져옴
    SPEAKER_V_PATH=$(ls "$VIDEO_DIR/${prefix}utt${prev_utt}"_*.mp4 2>/dev/null | head -n 1)

    # Speaker 파일이 없으면 건너뜀
    if [ -z "$SPEAKER_V_PATH" ]; then
        echo "Warning: Speaker file for ${dia_info} (utt${prev_utt}) not found. Skipping..."
        continue
    fi

    echo "------------------------------------------------"
    echo "Processing: $dia_info"
    echo "Speaker: $SPEAKER_V_PATH"
    echo "Listener Tgt: $LISTENER_V_PATH"
    echo "------------------------------------------------"

    # 4. 파이썬 명령어 실행
    python demo_listener_pt_correct.py \
        --ckpt /home/yjcho/EmpaTalk/listener_exp/v1/checkpoint/238000.pt \
        --pretrained /home/yjcho/EmpaTalk/ckpts/ckpts/EDTalk.pt \
        --speaker "$SPEAKER_V_PATH" \
        --listener_tgt "$LISTENER_V_PATH" \
        --listener_ref "$REF_IMG" \
        --listener_audio "$LISTENER_A_PATH" \
        --listener_pt "$pt_file" \
        --audio2lip_ckpt /home/yjcho/EmpaTalk/ckpts/ckpts/Audio2Lip.pt \
        --out "$OUT_DIR/${dia_info}.mp4"


done