-- Schéma PostgreSQL pour le chatbot RAG juridique et administratif
-- Version locale SANS pgvector.
-- Les embeddings sont stockés dans une colonne DOUBLE PRECISION[].
-- La similarité cosinus est calculée côté Python.

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    page INTEGER,
    chunk_id INTEGER,
    content TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING GIN(metadata);
