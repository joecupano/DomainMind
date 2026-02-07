-- Audit Database Initialization Script
-- Chain of Custody and Compliance Tracking

-- Create audit events table
CREATE TABLE IF NOT EXISTS audit_events (
    id SERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address INET,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    action VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    details JSONB,
    hash VARCHAR(64) NOT NULL,  -- SHA256 hash for integrity
    previous_hash VARCHAR(64),   -- Chain previous event
    CONSTRAINT valid_action CHECK (action IN ('CREATE', 'READ', 'UPDATE', 'DELETE', 'EXECUTE', 'ACCESS', 'EXPORT', 'LOGIN', 'LOGOUT')),
    CONSTRAINT valid_status CHECK (status IN ('SUCCESS', 'FAILURE', 'DENIED', 'ERROR'))
);

-- Create index for faster queries
CREATE INDEX idx_audit_timestamp ON audit_events(timestamp DESC);
CREATE INDEX idx_audit_event_type ON audit_events(event_type);
CREATE INDEX idx_audit_user ON audit_events(user_id);
CREATE INDEX idx_audit_session ON audit_events(session_id);
CREATE INDEX idx_audit_resource ON audit_events(resource_type, resource_id);

-- Create table for model interactions (LLM specific)
CREATE TABLE IF NOT EXISTS model_interactions (
    id SERIAL PRIMARY KEY,
    interaction_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    prompt_hash VARCHAR(64) NOT NULL,  -- Hash of prompt for privacy
    prompt_length INTEGER NOT NULL,
    response_hash VARCHAR(64) NOT NULL,  -- Hash of response
    response_length INTEGER NOT NULL,
    tokens_used INTEGER,
    latency_ms INTEGER,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    context_documents JSONB,  -- Which documents were used in RAG
    metadata JSONB,
    CONSTRAINT valid_model_status CHECK (status IN ('SUCCESS', 'FAILURE', 'TIMEOUT', 'REJECTED'))
);

CREATE INDEX idx_model_timestamp ON model_interactions(timestamp DESC);
CREATE INDEX idx_model_session ON model_interactions(session_id);
CREATE INDEX idx_model_name ON model_interactions(model_name);

-- Create table for document access tracking
CREATE TABLE IF NOT EXISTS document_access (
    id SERIAL PRIMARY KEY,
    access_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    document_id VARCHAR(255) NOT NULL,
    document_name VARCHAR(500),
    document_hash VARCHAR(64) NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    access_type VARCHAR(50) NOT NULL,
    purpose TEXT,
    granted BOOLEAN NOT NULL,
    reason TEXT,
    CONSTRAINT valid_access_type CHECK (access_type IN ('READ', 'EMBED', 'INDEX', 'EXPORT', 'DELETE'))
);

CREATE INDEX idx_doc_access_timestamp ON document_access(timestamp DESC);
CREATE INDEX idx_doc_id ON document_access(document_id);
CREATE INDEX idx_doc_session ON document_access(session_id);

-- Create table for data exports/extractions (compliance)
CREATE TABLE IF NOT EXISTS data_exports (
    id SERIAL PRIMARY KEY,
    export_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    export_type VARCHAR(100) NOT NULL,
    data_classification VARCHAR(50),
    record_count INTEGER,
    size_bytes BIGINT,
    destination VARCHAR(500),
    justification TEXT,
    approved_by VARCHAR(255),
    status VARCHAR(50) NOT NULL,
    hash VARCHAR(64) NOT NULL,
    CONSTRAINT valid_export_status CHECK (status IN ('PENDING', 'APPROVED', 'COMPLETED', 'DENIED', 'FAILED'))
);

CREATE INDEX idx_exports_timestamp ON data_exports(timestamp DESC);
CREATE INDEX idx_exports_user ON data_exports(user_id);
CREATE INDEX idx_exports_status ON data_exports(status);

-- Create table for system changes (configuration, models, etc.)
CREATE TABLE IF NOT EXISTS system_changes (
    id SERIAL PRIMARY KEY,
    change_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    change_type VARCHAR(100) NOT NULL,
    component VARCHAR(100) NOT NULL,
    admin_id VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    before_state JSONB,
    after_state JSONB,
    rollback_available BOOLEAN DEFAULT false,
    rollback_script TEXT,
    approved_by VARCHAR(255),
    change_ticket VARCHAR(100),
    CONSTRAINT valid_change_type CHECK (change_type IN ('CONFIG', 'MODEL', 'SECURITY', 'DEPLOYMENT', 'PATCH', 'ROLLBACK'))
);

CREATE INDEX idx_system_changes_timestamp ON system_changes(timestamp DESC);
CREATE INDEX idx_system_changes_type ON system_changes(change_type);
CREATE INDEX idx_system_changes_component ON system_changes(component);

-- Create table for security events
CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    event_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    severity VARCHAR(20) NOT NULL,
    event_category VARCHAR(100) NOT NULL,
    source_ip INET,
    user_id VARCHAR(255),
    description TEXT NOT NULL,
    technical_details JSONB,
    mitigated BOOLEAN DEFAULT false,
    mitigation_details TEXT,
    false_positive BOOLEAN DEFAULT false,
    CONSTRAINT valid_severity CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

CREATE INDEX idx_security_timestamp ON security_events(timestamp DESC);
CREATE INDEX idx_security_severity ON security_events(severity);
CREATE INDEX idx_security_category ON security_events(event_category);
CREATE INDEX idx_security_mitigated ON security_events(mitigated);

-- Create read-only user for audit queries (compliance officer)
CREATE USER audit_reader WITH PASSWORD 'read_only_audit_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO audit_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO audit_reader;

-- Prevent deletion of audit records
CREATE OR REPLACE FUNCTION prevent_audit_deletion()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Deletion from audit tables is not allowed. Archive old records instead.';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER prevent_audit_events_deletion
BEFORE DELETE ON audit_events
FOR EACH ROW EXECUTE FUNCTION prevent_audit_deletion();

CREATE TRIGGER prevent_model_interactions_deletion
BEFORE DELETE ON model_interactions
FOR EACH ROW EXECUTE FUNCTION prevent_audit_deletion();

CREATE TRIGGER prevent_document_access_deletion
BEFORE DELETE ON document_access
FOR EACH ROW EXECUTE FUNCTION prevent_audit_deletion();

CREATE TRIGGER prevent_data_exports_deletion
BEFORE DELETE ON data_exports
FOR EACH ROW EXECUTE FUNCTION prevent_audit_deletion();

CREATE TRIGGER prevent_system_changes_deletion
BEFORE DELETE ON system_changes
FOR EACH ROW EXECUTE FUNCTION prevent_audit_deletion();

CREATE TRIGGER prevent_security_events_deletion
BEFORE DELETE ON security_events
FOR EACH ROW EXECUTE FUNCTION prevent_audit_deletion();

-- Create function to chain audit events (blockchain-style)
CREATE OR REPLACE FUNCTION chain_audit_event()
RETURNS TRIGGER AS $$
DECLARE
    last_hash VARCHAR(64);
BEGIN
    -- Get the hash of the previous event
    SELECT hash INTO last_hash
    FROM audit_events
    ORDER BY timestamp DESC, id DESC
    LIMIT 1;
    
    -- Set the previous_hash for this event
    NEW.previous_hash := last_hash;
    
    -- Calculate hash of current event (simplified - in production use proper crypto)
    NEW.hash := encode(sha256(
        (NEW.event_id || NEW.timestamp || NEW.event_type || 
         COALESCE(NEW.user_id, '') || COALESCE(NEW.action, '') || 
         COALESCE(NEW.previous_hash, ''))::bytea
    ), 'hex');
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER chain_audit_events_trigger
BEFORE INSERT ON audit_events
FOR EACH ROW EXECUTE FUNCTION chain_audit_event();

-- Create archive tables for old audit records (>1 year)
CREATE TABLE IF NOT EXISTS audit_events_archive (LIKE audit_events INCLUDING ALL);
CREATE TABLE IF NOT EXISTS model_interactions_archive (LIKE model_interactions INCLUDING ALL);
CREATE TABLE IF NOT EXISTS document_access_archive (LIKE document_access INCLUDING ALL);

-- Grant appropriate permissions
GRANT INSERT ON audit_events, model_interactions, document_access, 
                data_exports, system_changes, security_events TO audit_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO audit_user;

-- Create views for compliance reporting
CREATE OR REPLACE VIEW recent_security_events AS
SELECT * FROM security_events
WHERE timestamp > NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW high_value_exports AS
SELECT * FROM data_exports
WHERE status = 'COMPLETED' 
  AND (size_bytes > 1000000 OR record_count > 1000)
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW user_activity_summary AS
SELECT 
    user_id,
    DATE(timestamp) as activity_date,
    COUNT(*) as total_events,
    COUNT(DISTINCT session_id) as sessions,
    COUNT(DISTINCT event_type) as event_types,
    SUM(CASE WHEN status = 'FAILURE' THEN 1 ELSE 0 END) as failures
FROM audit_events
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY user_id, DATE(timestamp)
ORDER BY activity_date DESC, total_events DESC;

GRANT SELECT ON recent_security_events, high_value_exports, user_activity_summary TO audit_reader;

-- Insert initial system event
INSERT INTO system_changes (
    change_type, component, admin_id, description, after_state, approved_by
) VALUES (
    'DEPLOYMENT', 'AUDIT_SYSTEM', 'system', 
    'Initial audit system deployment with chain of custody tracking',
    '{"version": "1.0.0", "features": ["audit_logging", "chain_of_custody", "compliance_tracking"]}'::jsonb,
    'system'
);

-- Create notification function for critical security events
CREATE OR REPLACE FUNCTION notify_critical_security_event()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.severity = 'CRITICAL' THEN
        -- In production, integrate with alerting system
        RAISE NOTICE 'CRITICAL SECURITY EVENT: %', NEW.description;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER notify_critical_events
AFTER INSERT ON security_events
FOR EACH ROW
WHEN (NEW.severity = 'CRITICAL')
EXECUTE FUNCTION notify_critical_security_event();
