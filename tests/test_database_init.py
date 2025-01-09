import threading
from unittest.mock import Mock, patch

import pytest
from flask import Flask
from celery import Celery
from celery.signals import worker_process_init

from seer.db import db, initialize_database
from celery_app.app import init_celery_app
from seer.configuration import AppConfig


def test_duplicate_db_initialization():
    """Test that database initialization is protected against duplicates across processes."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

    # First initialization - should succeed
    initialize_database(config=Mock(spec=AppConfig), app=app)

    # Second initialization - should not raise error due to protection
    initialize_database(config=Mock(spec=AppConfig), app=app)


def test_celery_worker_db_initialization():
    """Test that Celery worker initialization properly handles database setup."""
    celery_app = Celery('test_app')
    
    # Mock the app context
    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

    with patch('celery_app.app.has_app_context') as mock_has_context:
        mock_has_context.return_value = True

        # Simulate worker process initialization
        worker_process_init.send(sender=None)
        
        # First initialization through Celery worker
        init_celery_app(sender=celery_app)

        # Second initialization attempt should not raise an error
        # due to the process-safe initialization
        init_celery_app(sender=celery_app)


def test_concurrent_db_initialization():
    """Test that concurrent database initialization is handled properly within a process."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

    success_count = 0
    error_count = 0
    thread_count = 10

    def init_db():
        nonlocal success_count, error_count
        try:
            initialize_database(config=Mock(spec=AppConfig), app=app)
            success_count += 1
        except RuntimeError:
            error_count += 1

    threads = [threading.Thread(target=init_db) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Only one initialization should succeed
    assert success_count == 1
    assert error_count == thread_count - 1