import os
import tarfile
import glob
from tqdm import tqdm
from whisper_model import WhisperASR
from marian_translator import MarianTranslator
import sacrebleu

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TAR_FILE_PATH = os.path.join(CURRENT_DIR, "../Dataset/train-clean-100.tar.gz")
EXTRACT_DIR = os.path.join(CURRENT_DIR, "../Dataset/extracted_libri")

OUTPUT_FILE = os.path.join(CURRENT_DIR, "pipeline_results.txt")
REF_FILE = os.path.join(CURRENT_DIR, "references.txt")

def setup_dataset():
    """Giai nen dataset LibriSpeech neu chua co"""
    if not os.path.exists(EXTRACT_DIR):
        print(f"Dang giai nen: {os.path.abspath(TAR_FILE_PATH)}")
        try:
            with tarfile.open(TAR_FILE_PATH, "r:gz") as tar:
                tar.extractall(path=EXTRACT_DIR)
        except FileNotFoundError:
            print(f"Loi: Khong tim thay file dataset tai {os.path.abspath(TAR_FILE_PATH)}")
            exit()
    return os.path.join(EXTRACT_DIR, "LibriSpeech/train-clean-100")

def get_audio_files(root_dir, limit=1):
    """Lay danh sach file audio"""
    # Chọn file
    files = glob.glob(os.path.join(root_dir, "**/*.flac"), recursive=True)
    return files[:limit]

def load_references(ref_path):
    """Doc file references.txt va tra ve Dictionary {filename: text}"""
    refs = {}
    if not os.path.exists(ref_path):
        print("Khong tim thay file references.txt")
        return refs
    
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                filename = parts[0].strip()
                text = parts[1].strip()
                refs[filename] = text
    return refs

def main():
    # Setup Data
    data_root = setup_dataset()
    #audio_files = get_audio_files(data_root, limit=2) 

    target_file = r"D:\BK\Subject\HK252\DATN\Overall\Dataset\extracted_libri\LibriSpeech\train-clean-100\103\1240\103-1240-0008.flac"
    audio_files = [target_file]

    references_dict = load_references(REF_FILE)

    if not audio_files:
        print("Khong tim thay file audio.")
        exit()
        
    print(f"Tim thay {len(audio_files)} file audio de xu ly.")

    # Load Models
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    print("1. Loading Whisper (ASR)")
    asr_model = WhisperASR(model_name="base.en", device=device)
    
    print("2. Loading MarianMT (Translation)")
    mt_model = MarianTranslator(device=device)

    results = []

    # Pipeline Loop
    print("\nBat dau chay Pipeline: Audio -> En -> Vi")
    
    for audio_path in tqdm(audio_files):
        try:
            filename = os.path.basename(audio_path)
            
            # ASR
            en_text = asr_model.transcribe(audio_path).strip()
            
            # MT
            vi_text = mt_model.translate([en_text])[0]
            
            # Tính BLEU
            bleu_score = "N/A"
            ref_text = "N/A"
            
            if filename in references_dict:
                ref_text = references_dict[filename]
                # Tính Sentence BLEU
                score = sacrebleu.sentence_bleu(vi_text, [ref_text])
                bleu_score = f"{score.score:.2f}"

            results.append({
                "Filename": filename,
                "En": en_text,
                "Vi_Hyp": vi_text,    # output
                "Vi_Ref": ref_text,   # expect
                "BLEU": bleu_score
            })
            
        except Exception as e:
            print(f"Loi file {audio_path}: {e}")

    # Xuất file
    if results:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for item in results:
                f.write(f"Filename: {item['Filename']}\n")
                f.write(f"ASR (En): {item['En']}\n")
                f.write(f"MT  (output): {item['Vi_Hyp']}\n")
                f.write(f"REF (expect): {item['Vi_Ref']}\n")
                f.write(f"BLEU Score: {item['BLEU']}\n")
                f.write("-" * 50 + "\n")
        
        print(f"\nDa luu ket qua tai: {OUTPUT_FILE}")
        print("\n--- Noi dung file ---")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("Khong co ket qua nao duoc tao")

if __name__ == "__main__":
    main()