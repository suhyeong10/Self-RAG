# Self-RAG Implementation

Self-RAG (Self-Reflective Retrieval-Augmented Generation) 시스템의 구현입니다. 이 프로젝트는 LangGraph를 사용하여 Self-RAG 워크플로우를 구현합니다.

## 🚀 주요 기능

- **Self-RAG 모델**: `selfrag/selfrag_llama2_7b` 모델 사용
- **LangGraph 워크플로우**: 조건부 검색 및 생성
- **Reflection Tokens**: 검색 필요성, 관련성, 지원성, 유용성 평가
- **품질 검증**: 자동 품질 체크 및 재시도 메커니즘
- **PDF 문서 검색**: FAISS 벡터 데이터베이스 기반 검색

## 📋 요구사항

- Python 3.8+
- CUDA 지원 GPU (권장)
- 최소 16GB RAM

## 🛠️ 설치

```bash
# 저장소 클론
git clone <repository-url>
cd rag

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 🏃‍♂️ 사용법

### 기본 사용법

```python
from rag import self_rag

# Self-RAG 실행
result = self_rag("What is FaithfulRAG?")
print(result)
```

### 결과 형식

```python
{
    "query": "What is FaithfulRAG?",
    "response": "FaithfulRAG is...",
    "retrieval_needed": True,
    "retrieved_docs": [...],
    "reflection_tokens": ["retrieval", "relevant", "fully_supported", "utility_4"],
    "quality_metrics": {
        "is_relevant": True,
        "is_supported": True,
        "is_useful": 4
    },
    "retries": 0
}
```

## 🔧 설정

### 모델 설정
- 기본 모델: `selfrag/selfrag_llama2_7b`
- Transformers와 LangChain HuggingFace를 사용하여 추론

### 문서 검색 설정
- PDF 파일 경로: `dataset/pdf/`
- 벡터 데이터베이스 저장 경로: `dataset/`
- 임베딩 모델: `intfloat/multilingual-e5-base`

## 📊 Self-RAG 워크플로우

```
1. 쿼리 입력
   ↓
2. 검색 필요성 판단 (Retrieve 토큰)
   ↓
3. 검색이 필요한 경우:
   - 문서 검색
   - 문서와 함께 답변 생성
   - 관련성 평가 (ISREL)
   - 지원성 평가 (ISSUP)
   - 유용성 평가 (ISUSE)
   ↓
4. 검색이 불필요한 경우:
   - 직접 답변 생성
   - 유용성 평가 (ISUSE)
   ↓
5. 품질 검증
   - 기준 미달 시 재시도
   - 최대 3회 재시도
   ↓
6. 최종 답변 반환
```

## 🎯 Reflection Tokens

### Retrieve 토큰
- `[No Retrieval]`: 검색 불필요
- `[Retrieval]`: 검색 필요
- `[Continue to Use Evidence]`: 추가 증거 사용

### ISREL (Is Relevant) 토큰
- `[Relevant]`: 검색된 문서가 관련됨
- `[Irrelevant]`: 검색된 문서가 관련되지 않음

### ISSUP (Is Supported) 토큰
- `[Fully supported]`: 완전히 지원됨
- `[Partially supported]`: 부분적으로 지원됨
- `[No support / Contradictory]`: 지원되지 않음

### ISUSE (Is Useful) 토큰
- `[Utility:1-5]`: 5점 척도 유용성 평가

## 📁 프로젝트 구조

```
rag/
├── rag.py                 # 메인 Self-RAG 구현
├── requirements.txt       # Python 의존성
├── README.md             # 프로젝트 문서
├── .gitignore            # Git 무시 파일
└── dataset/              # 데이터셋 폴더
    ├── pdf/              # PDF 문서들
    ├── docs.pkl          # 처리된 문서
    └── database/         # FAISS 벡터 데이터베이스
```

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [Self-RAG 논문](https://arxiv.org/abs/2310.11511)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Hugging Face](https://huggingface.co/) 