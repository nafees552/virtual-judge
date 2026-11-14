"""Virtual Judge Flask application factory and global error handlers."""

from pathlib import Path

from flask import Flask, render_template, jsonify
from config import Config
from core.extensions import db, login_manager


def create_app():
    """Creates and configures Flask app instance with extensions and blueprints."""

    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)

    from routes import register_blueprints
    register_blueprints(app)

    with app.app_context():
        try:
            from core.models import User
            db.create_all()
            if not db.session.get(User, 1):
                db.session.add(User(id=1, username="Side A", email="side_a@virtualjudge.local", role="disputant"))
            if not db.session.get(User, 2):
                db.session.add(User(id=2, username="Side B", email="side_b@virtualjudge.local", role="disputant"))
            db.session.commit()
        except Exception:
            db.session.rollback()

    @app.errorhandler(404)
    def not_found(_err):
        """Returns custom 404 page."""
        return render_template("index.html", error_message="Page not found"), 404

    @app.errorhandler(500)
    def server_error(_err):
        """Returns JSON-friendly 500 response for easier debugging."""
        return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    with application.app_context():
        db.create_all()
    application.run(debug=True)
