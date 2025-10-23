"""BERT tokenizer와 text_to_sep_kata 동작 확인"""
import sys
sys.path.insert(0, r'C:\Users\Dael\Desktop\Dael\Project\L.U.N.A\luna-core')

from models.nlp import bert_models
from models.nlp.constants import Languages
from models.nlp.japanese.g2p import text_to_sep_kata, g2p

# 테스트 텍스트
test_text = "こんにちは、ルナです。テストを開始します。"

print("=" * 60)
print("1. 기본 텍스트 정보")
print("=" * 60)
print(f"원본 텍스트: {test_text}")
print(f"원본 텍스트 길이: {len(test_text)}")
print()

# text_to_sep_kata로 변환
sep_text, sep_kata = text_to_sep_kata(test_text, raise_yomi_error=False)
converted_text = "".join(sep_kata)

print("=" * 60)
print("2. text_to_sep_kata 변환 결과")
print("=" * 60)
print(f"sep_text: {sep_text}")
print(f"sep_kata: {sep_kata}")
print(f"converted_text: {converted_text}")
print(f"converted_text 길이: {len(converted_text)}")
print()

# Tokenizer 로드
tokenizer = bert_models.load_tokenizer(Languages.JP)
print("=" * 60)
print("3. 전체 텍스트 토큰화")
print("=" * 60)
print(f"Tokenizer 타입: {type(tokenizer).__name__}")
tokens = tokenizer.tokenize(converted_text)
print(f"토큰화 결과: {tokens}")
print(f"토큰 개수: {len(tokens)}")
print()

# BERT 모델의 입력 확인
inputs = tokenizer(converted_text, return_tensors="pt")
print(f"input_ids shape (CLS/SEP 포함): {inputs['input_ids'].shape}")
print(f"실제 토큰 수 (CLS/SEP 제외): {inputs['input_ids'].shape[1] - 2}")
print()

# g2p() 함수로 생성되는 word2ph 확인
print("=" * 60)
print("4. g2p() 함수의 word2ph 생성")
print("=" * 60)
phones, tones, word2ph = g2p(test_text, use_jp_extra=True, raise_yomi_error=False)
print(f"phones 개수: {len(phones)}")
print(f"word2ph: {word2ph}")
print(f"word2ph 길이: {len(word2ph)}")
print(f"word2ph 합계: {sum(word2ph)}")
print()

# 각 sep_kata 요소를 개별 토큰화
print("=" * 60)
print("5. 각 sep_kata 요소별 개별 토큰화 (g2p 방식)")
print("=" * 60)
from models.nlp.symbols import PUNCTUATIONS
for i, kata in enumerate(sep_kata):
    if kata not in PUNCTUATIONS:
        tokens_per_word = tokenizer.tokenize(kata)
        print(f"  '{kata}' ({sep_text[i]}) -> {tokens_per_word} (토큰 수: {len(tokens_per_word)})")
    else:
        print(f"  '{kata}' -> ['{kata}'] (구두점)")
print()

print("=" * 60)
print("6. 문제 진단")
print("=" * 60)
print(f"bert_feature.py에서 기대하는 길이:")
print(f"  - 변환된 텍스트 길이 + 2: {len(converted_text)} + 2 = {len(converted_text) + 2}")
print(f"g2p()에서 생성된 word2ph 길이: {len(word2ph)}")
print(f"BERT input_ids 길이 (CLS/SEP 포함): {inputs['input_ids'].shape[1]}")
print()
print(f"✗ 불일치: word2ph ({len(word2ph)}) != text length + 2 ({len(converted_text) + 2})")
print(f"✓ 일치: word2ph ({len(word2ph)}) == input_ids length ({inputs['input_ids'].shape[1]})")
