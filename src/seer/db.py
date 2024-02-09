import datetime

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.orm import sessionmaker

# Initialized in src/app.run
db: SQLAlchemy = SQLAlchemy()
migrate = Migrate(directory="src/migrations")
Session = sessionmaker()


# Why does mypy ignore type stubs???
class ProcessRequest(db.Model):  # type: ignore
    id = Column(Integer, primary_key=True)
    name = Column(String(128), index=True, unique=True, nullable=False)
    scheduled_for = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    scheduled_from = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    payload = Column(JSON, nullable=False)
