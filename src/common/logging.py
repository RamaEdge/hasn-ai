"""
Structured logging configuration for HASN-AI.

Provides JSON-structured logging with configurable levels and emoji indicators
for different system components.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Emoji indicators removed - no longer adding emojis to logs
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info",
                "exc_text", "stack_info"
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)
    
    def _get_emoji_indicator(self, record: logging.LogRecord) -> Optional[str]:
        """Get emoji indicator based on logger name or message content.
        
        Deprecated: Emoji indicators removed. This method returns None.
        """
        return None


def setup_logging(
    level: str = None,
    format_type: str = "json",
    enable_console: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Setup structured logging for the HASN-AI application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Defaults to INFO, can be overridden with DEBUG environment variable
        format_type: Output format ("json" or "text")
        enable_console: Whether to enable console logging
        log_file: Optional log file path
    """
    # Determine log level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
        if os.getenv("DEBUG", "").lower() in ("true", "1", "yes"):
            level = "DEBUG"
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Create formatter
    if format_type == "json":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Setup console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set level
    root_logger.setLevel(log_level)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        "level": level,
        "format": format_type,
        "console_enabled": enable_console,
        "log_file": log_file
    })


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience function for quick setup
def setup_default_logging() -> None:
    """Setup default logging configuration for HASN-AI."""
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format_type="json",
        enable_console=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    setup_logging(level="DEBUG")
    
    # Test different loggers
    brain_logger = get_logger("brain.network")
    api_logger = get_logger("api.routes")
    train_logger = get_logger("training.automated")
    config_logger = get_logger("config.setup")
    
    # Test different log levels and messages
    brain_logger.info("Brain network initialized with 100 neurons")
    brain_logger.debug("Neuron 42 spiked at time 1.23ms")
    
    api_logger.info("API endpoint /brain/process ready")
    api_logger.warning("Rate limit approaching for client 192.168.1.1")
    
    train_logger.info("Training completed successfully")
    train_logger.error("Failed to load training data from source")
    
    config_logger.info("Configuration loaded from environment")
    config_logger.critical("Critical system failure detected")
