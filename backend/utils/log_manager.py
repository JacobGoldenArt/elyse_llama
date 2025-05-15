import os
import logging
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

class LogManager:
    _instance: Optional['LogManager'] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_file: str = None, level: int = logging.INFO):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._loggers = {}
            self.default_level = level
            self.log_file = log_file
            self.console = Console(theme=Theme({
                "info": "blue",
                "warning": "yellow",
                "error": "red bold",
                "success": "green"
            }))
            self._initialized = True

    def _format_message(self, message: str, level: str) -> Panel:
        """Format a message with Rich styling based on log level"""
        return Panel(
            message,
            title=level.upper(),
            style=level.lower()
        )

    def info(self, message: str, logger_name: str = None):
        """Log an info message with Rich formatting"""
        self.console.print(self._format_message(message, "INFO"))
        if logger_name:
            self.get_logger(logger_name).info(message)

    def warning(self, message: str, logger_name: str = None):
        """Log a warning message with Rich formatting"""
        self.console.print(self._format_message(message, "WARNING"))
        if logger_name:
            self.get_logger(logger_name).warning(message)

    def error(self, message: str, logger_name: str = None):
        """Log an error message with Rich formatting"""
        self.console.print(self._format_message(message, "ERROR"))
        if logger_name:
            self.get_logger(logger_name).error(message)

    def success(self, message: str, logger_name: str = None):
        """Log a success message with Rich formatting"""
        self.console.print(self._format_message(message, "SUCCESS"))
        if logger_name:
            self.get_logger(logger_name).info(f"SUCCESS: {message}")

    def print_info(self, message: str):
        """Print an info message with nice formatting"""
        self.console.print(Panel(message, style="info"))
        self.info(message)

    def print_error(self, message: str):
        """Print an error message with nice formatting"""
        self.console.print(Panel(message, style="error"))
        self.error(message)

    def print_success(self, message: str):
        """Print a success message with nice formatting"""
        self.console.print(Panel(message, style="success"))
        self.success(message)

    def print_warning(self, message: str):
        """Print a warning message with nice formatting"""
        self.console.print(Panel(message, style="warning"))
        self.warning(message)

    @classmethod
    def get_logger(cls, name: str = __name__) -> logging.Logger:
        if cls._instance is None:
            cls(log_file=os.path.join(os.getcwd(), "app.log"), level=logging.DEBUG)
            
        if name not in cls._instance._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(cls._instance.default_level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler if log_file is specified
            if cls._instance.log_file:
                file_handler = logging.FileHandler(cls._instance.log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
            cls._instance._loggers[name] = logger
            
        return cls._instance._loggers[name]

    @classmethod
    def get_instance(cls) -> 'LogManager':
        """Get the singleton instance of LogManager"""
        if cls._instance is None:
            cls(log_file=os.path.join(os.getcwd(), "app.log"), level=logging.DEBUG)
        return cls._instance
