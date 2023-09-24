import API
from os import getenv
PORT = getenv('PORT')
if PORT is None:
    exit('Error: API server port is not specified\n')
API.create_app().run(host="0.0.0.0", port=int(PORT))

