"""Shared Flask extensions for VerdictBox."""

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "disputes.home"
login_manager.login_message_category = "warning"
