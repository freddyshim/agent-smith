-- Chat logs table for storing conversation history
CREATE TABLE IF NOT EXISTS chat_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL DEFAULT 'default',
    message_type VARCHAR(10) NOT NULL CHECK (message_type IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for efficient retrieval of recent messages by session
CREATE INDEX IF NOT EXISTS idx_chat_logs_session_timestamp ON chat_logs(session_id, timestamp DESC);

-- Index for full-text search on message content
CREATE INDEX IF NOT EXISTS idx_chat_logs_content_gin ON chat_logs USING gin(to_tsvector('english', content));