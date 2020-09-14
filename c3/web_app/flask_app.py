import flask

APP = flask.Flask(__name__)

@APP.route('/')
def print():
    return 'Fire risk analysis'

if __name__ == "__main__":
    APP.debug=True
    APP.run()