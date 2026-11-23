# AI Resume Screener — RAG

An AI-powered resume screening system built using Retrieval-Augmented Generation (RAG) to analyze candidate resumes against job requirements, retrieve relevant information, and generate context-aware candidate evaluations.

## Overview

Recruitment often involves manually reviewing numerous resumes to identify candidates whose qualifications, skills, and experience align with a job description.

This project explores how Large Language Models (LLMs) and Retrieval-Augmented Generation can assist in this process by extracting relevant information from resumes, retrieving contextually relevant content, and generating structured candidate assessments.

The system aims to make resume screening more efficient, consistent, and explainable by grounding its analysis in the information available within candidate resumes rather than relying entirely on the language model's general knowledge.

## Key Objectives

- Automate the initial resume screening process.
- Identify skills, qualifications, experience, and relevant projects from candidate resumes.
- Evaluate candidate profiles against job descriptions.
- Use retrieval-based context to ground AI-generated assessments.
- Reduce irrelevant or unsupported AI-generated conclusions.
- Evaluate the reliability, relevance, and performance of the screening system.

## Features

- **Resume Analysis:** Extract and analyze relevant candidate information from resumes.
- **Job Description Matching:** Compare candidate qualifications and skills against job requirements.
- **Retrieval-Augmented Generation:** Retrieve relevant resume information to provide context to the LLM during evaluation.
- **Context-Aware Assessment:** Generate candidate evaluations grounded in retrieved resume content.
- **AI-Powered Screening:** Use an LLM to interpret candidate information and generate assessments.
- **Docker Support:** Containerized application configuration for reproducible deployment and execution.

## System Architecture

The application follows a Retrieval-Augmented Generation pipeline:

```text
          Resume + Job Description
                    |
                    v
          Document Processing
                    |
                    v
          Text Extraction
                    |
                    v
          Chunking & Preparation
                    |
                    v
          Embedding Generation
                    |
                    v
          Vector Store / Retrieval
                    |
                    v
       Relevant Context Retrieval
                    |
                    v
        LLM-Based Candidate Analysis
                    |
                    v
       Structured Screening Output
```

### Pipeline Components

| Component | Responsibility |
|---|---|
| Document Processing | Process uploaded resume documents |
| Text Extraction | Extract textual information from resumes |
| Chunking | Divide documents into manageable text segments |
| Embeddings | Represent resume text as semantic vectors |
| Retrieval | Retrieve relevant resume information for a query |
| LLM | Analyze retrieved context and generate candidate evaluations |
| Application Layer | Coordinate the screening workflow and user interaction |

## How It Works

1. A resume and relevant job description are provided to the system.
2. Resume content is processed and converted into text.
3. The extracted text is divided into chunks suitable for retrieval.
4. Text chunks are transformed into embeddings and stored in a retrieval system.
5. The screening query is constructed using the job requirements.
6. Relevant resume information is retrieved based on the query.
7. The retrieved context is provided to the LLM.
8. The model generates a candidate assessment based on the available evidence.

## Evaluation

The system is currently being evaluated to understand the quality and reliability of its screening pipeline.

The evaluation focuses on the following dimensions:

### 1. Retrieval Quality

Assess whether the retrieval system identifies the most relevant resume information for a given job requirement.

Potential metrics:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Context relevance

### 2. Screening Accuracy

Evaluate whether the generated candidate assessment correctly reflects the candidate's actual qualifications, skills, and experience.

Areas of evaluation:

- Skill matching accuracy
- Experience relevance
- Qualification matching
- Identification of relevant projects
- Correct interpretation of candidate information

### 3. Groundedness

Determine whether generated assessments are supported by the retrieved resume context.

The evaluation should identify:

- Unsupported claims
- Hallucinated skills or experience
- Incorrect conclusions
- Missing evidence
- Misinterpretation of retrieved content

### 4. Relevance

Measure how closely the generated evaluation aligns with the job description and its requirements.

### 5. Consistency

Assess whether the system produces reasonably consistent evaluations when the same candidate and job description are evaluated repeatedly under equivalent conditions.

### 6. Performance

Measure operational performance, including:

- Processing time
- Retrieval latency
- LLM response time
- End-to-end screening time

### Evaluation Status

| Evaluation Dimension | Status |
|---|---|
| Retrieval Quality | Planned |
| Screening Accuracy | Planned |
| Groundedness | Planned |
| Relevance | Planned |
| Consistency | Planned |
| Performance | Planned |

Evaluation results and findings will be documented as testing progresses.

## Technology Stack

The project uses a Python-based application stack with containerization support.

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LLM | Natural language understanding and candidate assessment |
| RAG | Context-grounded generation |
| Embeddings | Semantic representation of resume content |
| Vector Retrieval | Retrieval of relevant resume information |
| Docker | Application containerization |
| Docker Compose | Container orchestration and environment management |

*Specific frameworks, models, embedding providers, and vector database implementations are documented as the architecture is finalized.*

## Project Structure

```text
ai-resume-screener-rag/
│
├── app/
│   └── Application source code
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- Git
- Docker and Docker Compose (for containerized execution)

### 1. Clone the Repository

```bash
git clone https://github.com/Noshaba-Fatima/ai-resume-screener-rag.git
```

Navigate into the project:

```bash
cd ai-resume-screener-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file if the application requires API keys or other configuration values.

```env
# Add required model/API credentials and configuration
```

Do not commit API keys, credentials, or sensitive candidate information to version control.

### 5. Run the Application

Use the project's configured entry point to start the application.

## Docker Deployment

The repository includes Docker configuration for running the application in a containerized environment.

Build and start the application:

```bash
docker compose up --build
```

To stop the running containers:

```bash
docker compose down
```

Refer to the Docker configuration for the application's exposed ports and environment variables.

## Limitations and Responsible Use

- AI-generated candidate assessments may contain inaccuracies.
- Retrieval quality depends on document quality, chunking, embeddings, and retrieval configuration.
- Resume formatting and document quality can affect text extraction.
- Candidate suitability cannot be fully determined from a resume alone.
- The system should not be used as the sole basis for hiring decisions.
- Human review is essential for validating generated assessments.
- Candidate information should be processed with appropriate privacy and security safeguards.
- The system should be evaluated for bias and fairness before being used in real recruitment workflows.

## Future Improvements

- Comprehensive retrieval evaluation using a benchmark dataset.
- Quantitative evaluation of screening accuracy and groundedness.
- Better document parsing for complex resume formats.
- Improved retrieval strategies and chunking techniques.
- Structured candidate evaluation with evidence references.
- Evaluation of hallucination rates.
- Candidate comparison and ranking capabilities.
- Monitoring of latency, token usage, and inference costs.
- Improved privacy and data handling.
- Automated testing and evaluation pipelines.

## Author

**Noshaba Fatima**

GitHub: [Noshaba-Fatima](https://github.com/Noshaba-Fatima)

## Repository

[AI Resume Screener — RAG](https://github.com/Noshaba-Fatima/ai-resume-screener-rag)
