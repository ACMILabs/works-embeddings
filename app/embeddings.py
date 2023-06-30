import os
import requests
from flask import Flask, render_template

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
XOS_API_ENDPOINT = os.getenv('XOS_API_ENDPOINT', None)
XOS_API_TOKEN = os.getenv('XOS_API_TOKEN', None)
XOS_RETRIES = int(os.getenv('XOS_RETRIES', '3'))
XOS_TIMEOUT = int(os.getenv('XOS_TIMEOUT', '60'))

application = Flask(__name__)


@application.route('/')
def home():
    """
    Works embeddings home page.
    """
    return render_template(
        'index.html',
    )


class XOSAPI():  # pylint: disable=too-few-public-methods
    """
    XOS private API interface.
    """
    def __init__(self):
        self.uri = XOS_API_ENDPOINT
        self.headers = {
            'Content-Type': 'application/json',
        }
        self.params = {
            'page_size': 10,
        }

    def get(self, resource, params=None):
        """
        Returns JSON for this resource.
        """
        endpoint = os.path.join(self.uri, f'{resource}/')
        if not params:
            params = self.params.copy()
        if XOS_API_TOKEN:
            self.headers['Authorization'] = f'Token {XOS_API_TOKEN}'
        retries = 0
        while retries < XOS_RETRIES:
            try:
                response = requests.get(
                    url=endpoint,
                    headers=self.headers,
                    params=params,
                    timeout=XOS_TIMEOUT,
                )
                response.raise_for_status()
                return response
            except (
                requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ) as exception:
                print(
                    f'ERROR: couldn\'t get {endpoint} with params: {params}, '
                    f'exception: {exception}... retrying',
                )
                retries += 1
                if retries == XOS_RETRIES:
                    raise exception
        return None


if __name__ == '__main__':
    application.run(
        host='0.0.0.0',
        port=8081,
        debug=DEBUG,
    )
