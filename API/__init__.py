from flask import Flask, make_response, Response, jsonify
#from pony.flask import Pony
def create_app(config_path: str = ''):
    app = Flask(__name__)
    app.config.from_pyfile(config_path, silent=True)
    
    #Here regisnter blueprints
    from . import bot_bp
    app.register_blueprint(bot_bp.bp)
   
    from . import files_bp
    app.register_blueprint(files_bp.bp)
    @app.route('/')
    def index():
        return  Response(status=200)
    #Pony(app)
    return app
