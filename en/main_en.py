import argparse
import os
import tarfile
import glob
from tqdm import tqdm
import sacrebleu

from whisper_model import WhisperASR
from dataset import normalize_text
from metrics import (
    compute_basic_metrics,
    compute_bert_score,
    compute_semantic_error_rate
)
from lm import refine_transcript

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TAR_FILE_PATH = os.path.join(CURRENT_DIR, "../Dataset/train-clean-100.tar.gz")
EXTRACT_DIR = os.path.join(CURRENT_DIR, "../Dataset/extracted_libri")

LIBRI_ROOT = os.path.join(
    EXTRACT_DIR, "LibriSpeech", "train-clean-100"
)

OUTPUT_FILE = "transcript.txt"

# Dataset Utilities
def setup_librispeech():
    if not os.path.exists(LIBRI_ROOT):
        print(f"Extracting LibriSpeech from {TAR_FILE_PATH}")
        with tarfile.open(TAR_FILE_PATH, "r:gz") as tar:
            tar.extractall(EXTRACT_DIR)
    return LIBRI_ROOT

def load_transcripts(root_dir):
    """
    Load LibriSpeech *.trans.txt
    Returns: {utt_id: text}
    """
    transcripts = {}
    for trans_file in glob.glob(
        os.path.join(root_dir, "**/*.trans.txt"),
        recursive=True
    ):
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, text = line.strip().split(" ", 1)
                transcripts[utt_id] = text
    return transcripts


def get_audio_files(root_dir, limit=None):
    files = glob.glob(
        os.path.join(root_dir, "**/*.flac"),
        recursive=True
    )
    return files[:limit] if limit else files

# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="base.en")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of audio files (0 = all)")
    return parser.parse_args()


# Main Pipeline

def main():
    args = parse_args()

    # Setup dataset
    data_root = setup_librispeech()
    transcripts = load_transcripts(data_root)
    audio_files = get_audio_files(
        data_root,
        limit=args.limit if args.limit > 0 else None
    )

    print(f"Found {len(audio_files)} audio files")

    # Device
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    # Models
    print("1️⃣ Loading Whisper ASR")
    asr_model = WhisperASR(args.model_name, device=device)

    asr_refs, asr_hyps = [], []

    results = []

    print("\n🚀 Running ASR pipeline")

    for audio_path in tqdm(audio_files):
        try:
            utt_id = os.path.splitext(os.path.basename(audio_path))[0]

            if utt_id not in transcripts:
                continue

            # ===== ASR =====
            en_hyp = asr_model.transcribe(audio_path)
            en_ref = transcripts[utt_id]

            asr_refs.append(normalize_text(en_ref))
            asr_hyps.append(normalize_text(en_hyp))

            results.append({
                "utt_id": utt_id,
                "en_ref": en_ref,
                "en_hyp": en_hyp,
            })

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Evaluation

    print("\n===== ASR Evaluation (Whisper) =====")
    basic = compute_basic_metrics(asr_refs, asr_hyps)
    for k, v in basic.items():
        print(f"{k}: {v:.4f}")

    print(f"BERTScore-F1: {compute_bert_score(asr_refs, asr_hyps):.4f}")
    print(f"Semantic Error Rate: {compute_semantic_error_rate(asr_refs, asr_hyps):.4f}")

    # Save results

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Utterance: {r['utt_id']}\n")
            f.write(f"ASR Ref (EN): {r['en_ref']}\n")
            f.write(f"ASR Hyp (EN): {r['en_hyp']}\n")
            f.write("-" * 50 + "\n")

    print(f"\n✅ Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
