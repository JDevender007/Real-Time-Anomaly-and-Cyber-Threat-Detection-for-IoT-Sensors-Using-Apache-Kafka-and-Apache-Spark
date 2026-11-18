"""Custom exception classes."""
from __future__ import annotations


class IoTSecurityError(Exception):
    """Base exception for IoT Security platform."""


class DatasetLoadError(IoTSecurityError):
    """Raised when a dataset cannot be loaded."""


class DatasetValidationError(IoTSecurityError):
    """Raised when a dataset fails validation."""


class ModelLoadError(IoTSecurityError):
    """Raised when a model cannot be loaded."""


class ModelTrainingError(IoTSecurityError):
    """Raised when model training fails."""


class KafkaConnectionError(IoTSecurityError):
    """Raised when Kafka connection fails."""


class StreamingProcessingError(IoTSecurityError):
    """Raised when streaming processing fails."""


class DatabaseConnectionError(IoTSecurityError):
    """Raised when database connection fails."""


class ConfigurationError(IoTSecurityError):
    """Raised when configuration is invalid."""


class SimulationError(IoTSecurityError):
    """Raised when simulation fails."""


class AlertGenerationError(IoTSecurityError):
    """Raised when alert generation fails."""


class FeatureEngineeringError(IoTSecurityError):
    """Raised when feature engineering fails."""
