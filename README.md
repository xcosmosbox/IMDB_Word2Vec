# IMDB Full-Stack Recommendation Platform

This README reflects the current state of the repository, which has evolved from a single Word2Vec experiment into a full-stack recommendation platform. The system now spans data ingestion and feature engineering, deep learning based generative ranking, LLM-assisted cold start, model serving, and multiple user-facing applications. The original README from the early Word2Vec phase is preserved at the end of this file for historical reference.

## Executive Overview

The repository is organized into four major pillars that together form an end-to-end recommendation system. First, `recommend-system/` provides production-style backend services in Go, including APIs for users, items, and recommendation delivery. It integrates relational storage (PostgreSQL with vector support), Redis caching, and Milvus for vector search, and it is designed for high throughput and low latency. Second, the algorithm stack in `recommend-system/algorithm/` implements a Unified Generative Transformer (UGT) with a semantic ID representation, a multi-stage training pipeline, and an export path to ONNX, TensorRT, and Triton for inference. Third, `imdb_word2vec/` contains a self-contained Python pipeline that builds a rich IMDb embedding space via feature engineering, autoencoder fusion, and Word2Vec training. Fourth, user experiences are provided by the Streamlit analytics dashboard in `web/` and the Vue 3 user application in `recommend-system/frontend/user-app/` (plus an admin console scaffold).

From a system perspective, data flows from collection and ETL to tokenization and feature engineering, then into deep learning training and export. Online services are built on top of a layered storage and retrieval stack, and LLM-driven cold start injects semantic priors for new users and new items. Observability, security, and compliance are first-class modules, with structured logging, monitoring dashboards, IAM scaffolding, and AI safety components. Planned functionality extends the semantic ID space to multi-modal signals, introduces richer LLM-guided preference bootstrapping, and deepens the coupling between the Word2Vec embedding space and the UGT semantic ID hierarchy.

## System Map

The system can be viewed as a pipeline that connects data engineering, model training, and online serving:

```
IMDb data + user events
  -> data-pipeline (collectors, ETL, feature engineering)
  -> semantic ID encoding + feature fusion
  -> UGT training (multi-stage)
  -> model export (ONNX/TensorRT/Triton)
  -> Go services (recommend, user, item)
  -> frontends (Streamlit analytics, Vue user app)
```

Key modules and their responsibilities:

- `recommend-system/cmd/`: Go service entrypoints (`recommend-service`, `user-service`, `item-service`).
- `recommend-system/internal/`: core application layers (model, repository, service, middleware, cache, inference, gRPC).
- `recommend-system/algorithm/`: Python algorithm modules (semantic ID, encoder, decoder, training, serving).
- `recommend-system/data-pipeline/`: collectors, ETL, feature-store, data quality, governance.
- `imdb_word2vec/`: IMDb data pipeline and Word2Vec embedding training.
- `web/`: Streamlit visualization and exploration tools for embeddings and recommendations.
- `recommend-system/frontend/user-app/`: Vue 3 UI with Pinia state, Vitest tests, and reusable components.
- `recommend-system/frontend/admin/`: admin UI scaffold and analytics views.
- `recommend-system/devops/` and `recommend-system/security/`: deployment, monitoring, security controls, and compliance utilities.

In addition to the core path, the repository contains substantial tooling around observability and governance: logging pipelines, monitoring dashboards, test suites for infrastructure configuration, and policies for IAM, data privacy, and AI safety.

## Algorithms and LLM Deep Dive

### Semantic IDs and Representation Learning

At the core of the algorithm stack is a semantic ID system that converts high-dimensional item descriptions into hierarchical discrete tokens. Each item is encoded into a three-level semantic ID (L1, L2, L3), where L1 captures coarse category, L2 represents finer attributes, and L3 differentiates instances. This representation is inspired by RQ-VAE style quantization and multi-codebook discretization, enabling large-scale catalog compression without losing semantic richness. Semantic IDs are stored and indexed in databases (including Postgres and Milvus) and act as the fundamental unit for modeling, retrieval, and generation.

The semantic ID hierarchy serves multiple purposes. It makes it possible to retrieve items by partial semantic matches (for example, L1 and L2 alignment when L3 is missing), supports hierarchical negative sampling in training, and creates a stable vocabulary for autoregressive decoding. It also provides a bridge between content features and user interaction sequences, letting the model reason about items in a compact, structured form. Planned work extends semantic IDs to multi-modal signals (text, images, metadata) and introduces cross-domain alignment objectives so that the same semantic structure can serve multiple verticals.

### UGT Encoder: Behavior Understanding

UGT treats recommendation as a sequence modeling problem. The encoder ingests event sequences that mix user actions, items, timestamps, and contextual tokens. Inputs are embedded through a unified representation: semantic embeddings for item IDs, positional embeddings to preserve order, token-type embeddings to distinguish user and item events, and time or context embeddings when available. The resulting representations are processed by stacked Transformer layers, yielding a user representation that captures long-range behavior patterns, short-term intent, and temporal dynamics.

Because the encoder operates on heterogeneous tokens, the implementation emphasizes robust embedding composition and masking. It is designed to support variable sequence lengths and dynamic padding, so both long histories and short sessions can be processed efficiently. This approach allows the model to learn rich user profiles without requiring separate models for different interaction types. It also supports a plug-in strategy for new token types, which is relevant for planned features such as richer context signals and cross-domain user journeys.

### UGT Decoder: Generative Recommendation

UGT departs from traditional candidate scoring by generating recommendations autoregressively as semantic ID sequences. Given the encoder output, the decoder predicts the next semantic ID triple (L1, L2, L3) step by step, using causal self-attention and cross-attention to incorporate user context. This design turns recommendation into a generative modeling task and enables the model to explore the recommendation space without a fixed candidate list. The generation process can be tuned for precision or diversity via beam search, diverse beam search, and nucleus sampling.

The decoder is aware of the semantic hierarchy and can enforce consistent generation across levels. For example, when L1 and L2 are generated, L3 is produced in a constrained space that respects the higher-level categories. This hierarchical generation makes the model more interpretable and robust to long-tail items. It also enables coarse-to-fine reasoning during inference, which is particularly useful for cold start and discovery scenarios.

### Mixture of Experts and Specialized Routing

The decoder includes Mixture-of-Experts (MoE) feed-forward layers, which route tokens to a subset of specialized expert networks. This structure is designed to improve capacity without uniformly increasing compute cost. Different experts can learn to handle different domains, user intents, or interaction modes, and a routing mechanism selects the most appropriate experts for each token. A load-balancing auxiliary loss helps avoid expert collapse and keeps capacity utilization healthy.

MoE allows the system to scale to larger item catalogs and more diverse behavior patterns while keeping latency manageable. It also lays the groundwork for future personalization strategies, where routing can be conditioned on user segments or contextual features. The planned roadmap includes stronger expert specialization objectives, dynamic routing based on real-time feedback, and improved gating mechanisms for multi-modal inputs.

### Training Pipeline and Loss Design

Training is organized as a multi-stage pipeline that mirrors the lifecycle of a modern generative system. The current design includes three stages: (1) Stage 1 pretraining with next token prediction (NTP) to learn basic sequence modeling; (2) Stage 2 multi-task fine-tuning with NTP plus contrastive objectives to align user and item representations; and (3) Stage 3 preference alignment with DPO-style losses to optimize ranking behavior. A unified loss aggregates NTP, contrastive, preference, and MoE balance components.

The training stack supports mixed precision, checkpointing, and distributed execution (DDP and DeepSpeed in design and configuration), making it feasible to train at scale. Data loaders and collators are built to handle streaming datasets and large-scale logs, while evaluation metrics track ranking quality and generation stability. Planned improvements include stronger offline-to-online consistency checks, automatic hyperparameter sweeps, and more explicit alignment between semantic ID reconstruction quality and downstream recommendation performance.

### Inference, Export, and Serving

Once trained, the model is exported through a pipeline that targets production performance. The algorithm serving module provides utilities to export PyTorch models to ONNX, optimize them with TensorRT, and deploy them in Triton Inference Server. The design prioritizes low latency and high throughput, with dynamic batching and configurable instance counts. This infrastructure enables the Go backend to call a high-performance inference service rather than running Python models inline.

The serving layer also includes benchmarking utilities and Triton configuration generators, which make it easier to validate performance targets and understand tradeoffs between precision (fp32, fp16, int8) and latency. As the system matures, the serving plan includes model versioning, canary deployment, and richer A/B experimentation support.

### LLM Integration and Cold Start

The repository includes a unified LLM client in `recommend-system/internal/llm/` that supports OpenAI, Azure OpenAI, Ollama, and custom HTTP backends. The client layer provides caching, retry policies, concurrency controls, and mock implementations to support both production and testing. This LLM layer is used by the cold start service in `recommend-system/internal/service/coldstart/`, which generates semantic priors for new users and items.

Cold start is handled through a multi-layer strategy. For new users, the system uses LLMs to infer initial preferences from sparse profile attributes and produces an initial recommendation list. For new items, the LLM analyzes titles, descriptions, and metadata to derive semantic features and embeddings. The system supports explainable recommendations by generating user-facing rationales derived from the LLM outputs. When LLM calls fail or are too expensive, a fallback strategy relies on rule-based or popularity-based suggestions, ensuring availability.

Planned features include richer prompt templates for different verticals, retrieval-augmented prompting based on the semantic ID hierarchy, and tighter integration of LLM embeddings with the UGT representation space. The roadmap also includes LLM-based user intent summarization, which could feed directly into the encoder as a contextual token stream.

### Evaluation, Safety, and Reliability

The algorithm stack is complemented by AI safety and security modules, such as prompt injection detection and content moderation tools. These components are designed to safeguard LLM usage and prevent prompt abuse or sensitive data leakage. The system also includes performance benchmarks for LLM clients and recommendation APIs, enabling consistent latency and throughput tracking.

Future improvements include richer evaluation suites that combine offline ranking metrics with online feedback loops, and safety evaluations that test prompt guardrails under adversarial conditions.

## Data and Feature Pipeline

The data layer is organized around structured collection, transformation, and feature management. In `recommend-system/data-pipeline/`, collectors ingest data streams and APIs, ETL steps normalize and validate raw events, and feature engineering transforms data into token-ready formats. A feature store architecture supports both offline storage (for training) and online storage (for inference), ensuring that feature definitions stay consistent between batch and real-time contexts. Data quality checks and governance tooling add schema validation, lineage tracking, and automated profiling.

This pipeline is designed to be modular, so new sources or feature transformations can be added without restructuring the entire system. Planned enhancements include stronger streaming support, automated anomaly detection in data quality monitors, and deeper integration with the semantic ID codebook management system.

## Word2Vec Pipeline (IMDb Embedding Space)

The `imdb_word2vec/` project remains a foundational component, now positioned as a complementary embedding track. It provides a full pipeline to download IMDb TSV data, clean and preprocess tables, perform feature engineering, fuse features with an autoencoder, and train Word2Vec embeddings using a skip-gram approach with negative sampling. The pipeline is fully modular and provides both a single-command "all" flow and granular step-by-step commands for download, preprocess, feature engineering, fusion, and Word2Vec training.

Artifacts include fused feature tables, vocabulary mappings, trained embeddings, and visualization outputs. The pipeline supports GPU acceleration when available and can be configured to run on a subset of the data for rapid experimentation. The resulting embeddings are used for similarity exploration and can be aligned with semantic IDs in future work, enabling a hybrid representation that combines semantic tokenization with dense vector geometry.

## Applications and UX

The Streamlit app in `web/` is a comprehensive visualization and exploration environment. It includes clustering analysis, nearest neighbor search, embedding arithmetic, dimensionality reduction comparisons (PCA, UMAP, t-SNE), and export tools for downstream usage. It acts as a diagnostic lens for the embedding space and helps validate the quality of learned representations.

The Vue 3 user application in `recommend-system/frontend/user-app/` provides a modern UI for search, recommendations, user history, and profile features. It uses Pinia for state management, Vite for tooling, and Vitest for testing. The admin console scaffold in `recommend-system/frontend/admin/` covers analytics dashboards and operational controls and can be expanded into a full internal tooling suite.

Planned UX improvements include richer recommendation explanations, interactive personalization controls, and more direct integration of model feedback into the UI (for example, showing semantic ID clusters or LLM-generated preference summaries).

## Operations, Security, and Compliance

Operational tooling is a first-class concern in this repository. Docker and Kubernetes deployment assets are provided under `recommend-system/deployments/` and `recommend-system/devops/kubernetes/`, while logging and monitoring assets are organized under `recommend-system/devops/logging/` and `recommend-system/devops/monitoring/`. Prometheus and Grafana dashboards provide visibility into service performance, and load-testing scripts support performance regression checks.

Security modules cover IAM scaffolding, WAF rules, API gateway signatures and rate limiting, secure headers middleware, data privacy utilities (encryption and masking), and compliance checks (GDPR-related tooling and audit logging). AI safety modules include prompt guard and content moderation utilities. These components are designed to be integrated into production deployments, with policy-driven enforcement and test coverage for critical configurations.

## Quickstart (Representative Commands)

Backend services (Go, from `recommend-system/`):

```
go mod download
make compose-up
make init-db
make run
```

Algorithm training (Python, from `recommend-system/algorithm/`):

```
pip install -r requirements.txt
python training/scripts/train_stage1.py --config configs/stage1.yaml
```

Word2Vec pipeline (from `imdb_word2vec/`):

```
python -m imdb_word2vec.cli all --subset-rows 100000 --max-rows 50000 --max-seq 50000
```

Streamlit dashboard (from `web/`):

```
pip install -r requirements.txt
streamlit run app.py
```

Vue user app (from `recommend-system/frontend/user-app/`):

```
npm install
npm run dev
```

## Roadmap and Planned Enhancements

The project roadmap emphasizes deeper algorithmic sophistication and stronger end-to-end integration. Planned directions include:

- Multi-modal semantic IDs that incorporate text, images, and structured metadata, with explicit cross-domain alignment objectives.
- A unified embedding space that bridges the Word2Vec pipeline and UGT tokenization, enabling dense vector retrieval alongside generative decoding.
- LLM-guided preference modeling at scale, including intent summarization, prompt templates per vertical, and retrieval-augmented prompts grounded in semantic IDs.
- Online learning and feedback loops to align model outputs with real-time user behavior, alongside safety guardrails for LLM usage.
- Improved inference orchestration with versioning, canary releases, and model fleet management for production-scale deployment.

These items are forward-looking and represent planned or in-progress directions rather than fully deployed features.

## Legacy README (Archived)

The content below is preserved from the original README for historical context.

# IMDB Movie Recommendation System By Word2Vec

This project is a movie recommendation system based on IMDB data, developed using Word2Vec and deep learning techniques. The goal is to embed movie features into a vector space and use these embeddings to recommend similar movies. The project also includes a visualization of movie embeddings using techniques such as PCA and UMAP.

## Project Structure

The workflow is as follows:

1. **Data Cleaning and Preprocessing**: Each table in the dataset undergoes extensive cleaning. Unnecessary columns are removed, missing values are handled, and key features are extracted.
2. **Feature Engineering**: Features such as movie genres, regions, types, and more are transformed into one-hot encoded vectors using a manually created vocabulary.
3. **Combining Data**: All the tables are merged into a final dataset, where each row represents a movie and its associated features.
4. **Embedding Features**: The final feature table is passed through a neural network, which is trained to learn the relationships between features. This process produces fused feature vectors that capture high-level information about each movie.
5. **Word2Vec Training**: Using the `Word2Vec` model, movie embeddings are further refined. By using a skip-gram approach with negative sampling, the model learns to place similar movies close together in the embedding space.
6. **Visualization**: The trained embeddings are visualized using PCA and UMAP, allowing us to see the relationships between movies in the latent space.

## PCA and UMAP Visualization

Below are the visualizations of the movie embeddings using PCA and UMAP:

### PCA Visualization

Due to limitations in Embedding Projector, only 2.4% of the data points are displayed.

![img_v3_02d7_b58791de-671a-47dd-8fa7-8c7a64061chu](https://raw.githubusercontent.com/xavierfrankland/PicRepo/master/uPic/TiIQyMimg_v3_02d7_b58791de-671a-47dd-8fa7-8c7a64061chu.jpg)

### UMAP Visualization

This UMAP result shows 5000 data points, which is approximately 0.2% of the total sample size.

![img_v3_02d7_055b15a9-59b3-4d91-9ddd-08b74030a9hu](https://raw.githubusercontent.com/xavierfrankland/PicRepo/master/uPic/yS2uKvimg_v3_02d7_055b15a9-59b3-4d91-9ddd-08b74030a9hu.jpg)

### Movie Embedding Exploration

The interactive visualization allows you to click on any point (a specific movie) and explore its neighbors—movies that are similar and recommended based on the embedding.

### Example of Movie Embedding Exploration

In this example, clicking on a movie reveals its neighboring points, representing other movies that are similar and recommended.

![img_v3_02d7_f6f00a71-a687-42f7-8f92-730e03e382hu](https://raw.githubusercontent.com/xavierfrankland/PicRepo/master/uPic/8kWCI2img_v3_02d7_f6f00a71-a687-42f7-8f92-730e03e382hu.jpg)

## Final Dataset

The final dataset consists of approximately 200,000 rows and 28 columns. The key feature table is created by one-hot encoding key attributes, like genres, movie types, and regions, using a manually constructed vocabulary.

### Final Table Size and Structure

The final dataset has undergone a detailed feature engineering process, with many unnecessary or redundant features removed. It contains crucial features like genres and regions, along with their one-hot encoded vectors, making it ideal for feeding into the neural network.

## Word2Vec Model

Once the final table was prepared, it was fed into a `Word2Vec` model. The purpose of this model was to further refine the relationships between movies by learning feature similarities and producing embeddings that could be used for movie recommendations.

### Word2Vec Results

The Word2Vec model provided high-dimensional movie embeddings. When examined through the Embedding Projector, the recommendations produced are highly accurate, with similar movies being placed close to each other in the vector space.
