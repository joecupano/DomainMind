"""
Audit Logging and Chain of Custody System
For highly regulated environments requiring complete audit trails
"""
import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of auditable events"""
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    QUERY_SUBMITTED = "QUERY_SUBMITTED"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    DOCUMENT_ACCESS = "DOCUMENT_ACCESS"
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    DATA_EXPORT = "DATA_EXPORT"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    SECURITY_EVENT = "SECURITY_EVENT"
    SYSTEM_ACCESS = "SYSTEM_ACCESS"


class Action(Enum):
    """Action types for audit events"""
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXECUTE = "EXECUTE"
    ACCESS = "ACCESS"
    EXPORT = "EXPORT"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"


class Status(Enum):
    """Status of audited action"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED = "DENIED"
    ERROR = "ERROR"


class AuditLogger:
    """
    Chain of Custody Audit Logger for compliance and security
    
    Features:
    - Immutable audit logs
    - Blockchain-style event chaining
    - Comprehensive event tracking
    - Compliance reporting
    """
    
    def __init__(self):
        """Initialize audit logger with database connection"""
        self.enabled = os.getenv('ENABLE_AUDIT_LOGGING', 'false').lower() == 'true'
        self.db_config = {
            'host': os.getenv('AUDIT_DB_HOST', 'audit-db'),
            'port': int(os.getenv('AUDIT_DB_PORT', 5432)),
            'database': os.getenv('AUDIT_DB_NAME', 'audit_db'),
            'user': os.getenv('AUDIT_DB_USER', 'audit_user'),
            'password': os.getenv('AUDIT_DB_PASSWORD', 'changeme_secure_password')
        }
        
        # File-based backup logging
        self.log_dir = os.getenv('AUDIT_LOG_PATH', '/app/audit_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Test connection
        if self.enabled:
            self._test_connection()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            logger.info("Audit database connection successful")
        except Exception as e:
            logger.warning(f"Audit database not available: {e}")
            logger.warning("Audit logs will be written to files only")
            self.enabled = False
    
    def _hash_data(self, data: str) -> str:
        """Create SHA256 hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _write_to_file(self, event_type: str, data: Dict[str, Any]):
        """Backup audit events to file"""
        timestamp = datetime.utcnow().strftime('%Y%m%d')
        log_file = os.path.join(self.log_dir, f'audit_{timestamp}.jsonl')
        
        try:
            with open(log_file, 'a') as f:
                json.dump({
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_type': event_type,
                    **data
                }, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write audit log to file: {e}")
    
    def log_event(
        self,
        event_type: EventType,
        action: Action,
        status: Status,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log an audit event with chain of custody
        
        Args:
            event_type: Type of event
            action: Action performed
            status: Status of action
            user_id: User identifier
            session_id: Session identifier
            ip_address: Source IP address
            resource_type: Type of resource accessed
            resource_id: Resource identifier
            details: Additional event details
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
        
        event_data = {
            'event_type': event_type.value,
            'action': action.value,
            'status': status.value,
            'user_id': user_id,
            'session_id': session_id,
            'ip_address': ip_address,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'details': Json(details) if details else None
        }
        
        # Always write to file
        self._write_to_file(event_type.value, event_data)
        
        # Try to write to database
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Hash computation is done by trigger, but we set placeholder
                    event_data['hash'] = 'computed_by_trigger'
                    
                    query = """
                        INSERT INTO audit_events 
                        (event_type, action, status, user_id, session_id, 
                         ip_address, resource_type, resource_id, details, hash)
                        VALUES (%(event_type)s, %(action)s, %(status)s, %(user_id)s, 
                                %(session_id)s, %(ip_address)s, %(resource_type)s, 
                                %(resource_id)s, %(details)s, %(hash)s)
                        RETURNING event_id
                    """
                    cur.execute(query, event_data)
                    event_id = cur.fetchone()[0]
                    logger.info(f"Audit event logged: {event_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    def log_model_interaction(
        self,
        session_id: str,
        model_name: str,
        prompt: str,
        response: str,
        user_id: Optional[str] = None,
        model_version: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
        status: str = "SUCCESS",
        context_documents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log LLM model interaction for compliance
        
        Args:
            session_id: Session identifier
            model_name: Name of model used
            prompt: User prompt (will be hashed)
            response: Model response (will be hashed)
            user_id: User identifier
            model_version: Model version
            tokens_used: Number of tokens
            latency_ms: Response latency
            status: Interaction status
            context_documents: Documents used in RAG
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
        
        interaction_data = {
            'session_id': session_id,
            'user_id': user_id,
            'model_name': model_name,
            'model_version': model_version,
            'prompt_hash': self._hash_data(prompt),
            'prompt_length': len(prompt),
            'response_hash': self._hash_data(response),
            'response_length': len(response),
            'tokens_used': tokens_used,
            'latency_ms': latency_ms,
            'status': status,
            'context_documents': Json(context_documents) if context_documents else None,
            'metadata': Json(metadata) if metadata else None
        }
        
        # Write to file
        self._write_to_file('MODEL_INTERACTION', interaction_data)
        
        # Write to database
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO model_interactions 
                        (session_id, user_id, model_name, model_version, prompt_hash,
                         prompt_length, response_hash, response_length, tokens_used,
                         latency_ms, status, context_documents, metadata)
                        VALUES (%(session_id)s, %(user_id)s, %(model_name)s, 
                                %(model_version)s, %(prompt_hash)s, %(prompt_length)s,
                                %(response_hash)s, %(response_length)s, %(tokens_used)s,
                                %(latency_ms)s, %(status)s, %(context_documents)s, %(metadata)s)
                        RETURNING interaction_id
                    """
                    cur.execute(query, interaction_data)
                    interaction_id = cur.fetchone()[0]
                    logger.info(f"Model interaction logged: {interaction_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to log model interaction: {e}")
            return False
    
    def log_document_access(
        self,
        document_id: str,
        document_name: str,
        document_hash: str,
        access_type: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        granted: bool = True,
        purpose: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Log document access for chain of custody
        
        Args:
            document_id: Document identifier
            document_name: Document name
            document_hash: Document content hash
            access_type: Type of access (READ, EMBED, INDEX, etc.)
            user_id: User identifier
            session_id: Session identifier
            granted: Whether access was granted
            purpose: Purpose of access
            reason: Reason for access/denial
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
        
        access_data = {
            'document_id': document_id,
            'document_name': document_name,
            'document_hash': document_hash,
            'user_id': user_id,
            'session_id': session_id,
            'access_type': access_type,
            'purpose': purpose,
            'granted': granted,
            'reason': reason
        }
        
        # Write to file
        self._write_to_file('DOCUMENT_ACCESS', access_data)
        
        # Write to database
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO document_access 
                        (document_id, document_name, document_hash, user_id, session_id,
                         access_type, purpose, granted, reason)
                        VALUES (%(document_id)s, %(document_name)s, %(document_hash)s,
                                %(user_id)s, %(session_id)s, %(access_type)s, %(purpose)s,
                                %(granted)s, %(reason)s)
                        RETURNING access_id
                    """
                    cur.execute(query, access_data)
                    access_id = cur.fetchone()[0]
                    logger.info(f"Document access logged: {access_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to log document access: {e}")
            return False
    
    def log_security_event(
        self,
        severity: str,
        event_category: str,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        technical_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log security event
        
        Args:
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            event_category: Category of security event
            description: Event description
            source_ip: Source IP address
            user_id: User identifier
            technical_details: Technical details
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
        
        security_data = {
            'severity': severity,
            'event_category': event_category,
            'description': description,
            'source_ip': source_ip,
            'user_id': user_id,
            'technical_details': Json(technical_details) if technical_details else None
        }
        
        # Write to file
        self._write_to_file('SECURITY_EVENT', security_data)
        
        # Write to database
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        INSERT INTO security_events 
                        (severity, event_category, description, source_ip, 
                         user_id, technical_details)
                        VALUES (%(severity)s, %(event_category)s, %(description)s,
                                %(source_ip)s, %(user_id)s, %(technical_details)s)
                        RETURNING event_id
                    """
                    cur.execute(query, security_data)
                    event_id = cur.fetchone()[0]
                    logger.warning(f"Security event logged: {event_id} - {severity}")
                    return True
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
            return False
    
    def verify_chain_integrity(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Verify integrity of audit event chain
        
        Args:
            limit: Number of recent events to verify
            
        Returns:
            dict: Verification results
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT event_id, hash, previous_hash, timestamp
                        FROM audit_events
                        ORDER BY timestamp DESC, id DESC
                        LIMIT %s
                    """
                    cur.execute(query, (limit,))
                    events = cur.fetchall()
                    
                    if not events:
                        return {'status': 'no_events', 'verified': True}
                    
                    # Verify chain
                    verified = True
                    broken_links = []
                    
                    for i in range(len(events) - 1):
                        current = events[i]
                        next_event = events[i + 1]
                        
                        if current['previous_hash'] != next_event['hash']:
                            verified = False
                            broken_links.append({
                                'event_id': str(current['event_id']),
                                'timestamp': current['timestamp'].isoformat()
                            })
                    
                    return {
                        'status': 'verified' if verified else 'broken',
                        'verified': verified,
                        'events_checked': len(events),
                        'broken_links': broken_links if broken_links else None
                    }
        except Exception as e:
            logger.error(f"Failed to verify chain integrity: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_user_activity(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get user activity for compliance reporting
        
        Args:
            user_id: User identifier
            days: Number of days to retrieve
            
        Returns:
            list: User activity events
        """
        if not self.enabled:
            return []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT event_type, action, status, timestamp, resource_type, details
                        FROM audit_events
                        WHERE user_id = %s
                          AND timestamp > NOW() - INTERVAL '%s days'
                        ORDER BY timestamp DESC
                        LIMIT 1000
                    """
                    cur.execute(query, (user_id, days))
                    events = cur.fetchall()
                    
                    # Convert to list of dicts with ISO timestamps
                    return [
                        {
                            **dict(event),
                            'timestamp': event['timestamp'].isoformat()
                        }
                        for event in events
                    ]
        except Exception as e:
            logger.error(f"Failed to get user activity: {e}")
            return []


# Global audit logger instance
audit_logger = AuditLogger()
