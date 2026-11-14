"""Blueprint registration for Virtual Judge routes."""

from routes.disputes import disputes_bp


def register_blueprints(app):
    """Registers application blueprints on the Flask app."""
    app.register_blueprint(disputes_bp)
