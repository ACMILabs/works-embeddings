import json
import os
import random
import time
import requests
from flask import Flask, jsonify, render_template, request, Response

import chromadb

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
REBUILD = os.getenv('REBUILD', 'false').lower() == 'true'
XOS_API_ENDPOINT = os.getenv('XOS_API_ENDPOINT', None)
XOS_RETRIES = int(os.getenv('XOS_RETRIES', '1'))
XOS_TIMEOUT = int(os.getenv('XOS_TIMEOUT', '60'))
DATABASE_PATH = os.getenv('DATABASE_PATH', '')
PORT = int(os.getenv('PORT', '8081'))
DEFAULT_TEMPLATE_JSON = os.getenv('DEFAULT_TEMPLATE_JSON', 'true').lower()
REFRESH_TIMEOUT = int(os.getenv('REFRESH_TIMEOUT', '0') or '0')

LENS_READER_TAPS_API = os.getenv('LENS_READER_TAPS_API', None)
AUTH_TOKEN = os.getenv('AUTH_TOKEN', None)

application = Flask(__name__)
CHROMA = None
SELECTED_WORK_ID = None
SUCCESSFUL_TAP = None


@application.template_filter('format_distance')
def format_distance(value):
    """
    Format the vector distance between embeddings to a percentage.
    """
    return int((1 - value) * 100)


@application.route('/')
def home():
    """
    Works embeddings home page.
    """
    results = None
    filtered_embeddings = CHROMA.embeddings

    json_response = request.args.get('json', DEFAULT_TEMPLATE_JSON).lower() == 'true'
    work = request.args.get('work', None)
    size = int(request.args.get('size', '11'))

    if work:
        global SELECTED_WORK_ID  # pylint: disable=global-statement
        SELECTED_WORK_ID = work
        for embedding in CHROMA.embeddings:
            if str(embedding['work']) == work:
                result = CHROMA.collection.get(
                    ids=embedding['id'],
                    include=['embeddings', 'documents'],
                )
                results = CHROMA.search(result['embeddings'][0], size)
        if results:
            filtered_embeddings = []
            for index, embedding_id in enumerate(results['ids'][0]):
                for embedding in CHROMA.embeddings:
                    if str(embedding['id']) == embedding_id:
                        embedding['distance'] = results['distances'][0][index]
                        filtered_embeddings.append(embedding)
    elif filtered_embeddings:
        filtered_embeddings = random.sample(filtered_embeddings, size)
        SELECTED_WORK_ID = None
    else:
        SELECTED_WORK_ID = None

    if not json_response:
        return render_template(
            'index.html',
            embeddings=filtered_embeddings,
            refresh_timeout=REFRESH_TIMEOUT,
        )

    return jsonify(filtered_embeddings)


def post_to_lens_reader(tap_data):
    """
    Post the tap back to the lens reader for queing to XOS.
    """
    global SUCCESSFUL_TAP  # pylint: disable=global-statement
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {AUTH_TOKEN}'
    }
    response = requests.post(
        LENS_READER_TAPS_API,
        headers=headers,
        data=json.dumps(tap_data),
        timeout=120,
    )
    if response.status_code == 201:
        SUCCESSFUL_TAP = 1
    else:
        SUCCESSFUL_TAP = 0
    print(f'Lens reader response: {response.json()}')


def event_stream():
    while True:
        global SUCCESSFUL_TAP  # pylint: disable=global-statement
        time.sleep(0.1)
        if (SELECTED_WORK_ID and SUCCESSFUL_TAP is not None) or SUCCESSFUL_TAP is not None:
            tap_event_message = f'data: {{ "tap_successful": {SUCCESSFUL_TAP} }}\n\n'
            SUCCESSFUL_TAP = None
            yield tap_event_message
        else:
            # Heartbeat to keep the connection alive
            yield ':\n\n'


if LENS_READER_TAPS_API:
    @application.route('/api/taps/', methods=['GET', 'POST'])
    def taps():
        """
        An API for adding the currently selected work -> label to a Tap.
        """
        global SUCCESSFUL_TAP  # pylint: disable=global-statement
        tap_data = {'label': None}
        if request.method == 'POST':
            tap_data = request.get_json()

        if SELECTED_WORK_ID:
            xos = XOSAPI()
            xos_response = xos.get(f'works/{SELECTED_WORK_ID}').json()
            tap_data['label'] = xos_response['labels'][0]
            post_to_lens_reader(tap_data)
            return jsonify(tap_data)
        SUCCESSFUL_TAP = 0
        return jsonify({'error': 'Please select a Work first before tapping.'}), 400


if LENS_READER_TAPS_API:
    @application.route('/api/tap-source/')
    def tap_source():
        headers = {
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream',
            'X-Accel-Buffering': 'no',
        }
        return Response(event_stream(), headers=headers, mimetype='text/event-stream')


if DEBUG:
    @application.route('/works/<work_id>/')
    def works(work_id):
        """
        Return an XOS Work.
        """
        xos = XOSAPI()
        return jsonify(xos.get(f'works/{work_id}').json())


class NoCollectionException(Exception):
    """
    This exception is raised if a collection doesn't exist.
    """


class Chroma():
    """
    Chroma vector database to interact with embeddings.
    """
    def __init__(self):
        self.client = chromadb.PersistentClient(path=f'{DATABASE_PATH}works_db')  # pylint: disable=no-member
        self.collection_name = 'works'
        self.collection = None
        self.embeddings = []

    def get_collection(self, name=None):
        """
        Get or create a collection of embeddings.
        """
        if not name:
            name = self.collection_name
        self.collection = self.client.get_or_create_collection(name=name)
        return self.collection

    def add_embedding(self, embeddings_json):
        """
        Add an embedding to Chroma from an individual XOS Embeddings JSON.
        """
        if not self.collection:
            raise NoCollectionException('Please create or load a collection')
        return self.collection.add(
            embeddings=[embeddings_json['data']['data'][0]['embedding']],
            documents=[str(embeddings_json['work'])],
            metadatas=[{'source': 'works'}],
            ids=[str(embeddings_json['id'])],
        )

    def add_pages_of_embeddings(self, pages=999999):
        """
        Add pages of XOS Embeddings API results to Chroma.
        """
        page = 1
        while page < (pages + 1):
            embeddings_json = XOSAPI().get(
                'embeddings',
                {'page': page, 'unpublished': 'false'},
            ).json()
            for embedding in embeddings_json['results']:
                self.add_embedding(embedding)
                self.embeddings.append({
                    'id': str(embedding['id']),
                    'work': str(embedding['work']),
                })
            print(f'Added {len(embeddings_json["results"])} from page {page}')
            page += 1
            if not embeddings_json['next']:
                print(f'Finished adding {len(CHROMA.embeddings)} items.')
                break

    def load_embeddings(self):
        """
        Load embeddings from the collection.
        """
        if not self.collection:
            self.get_collection()
        embedding_ids = self.collection.get()['ids']
        for index, embedding_id in enumerate(embedding_ids):
            result = self.collection.get(ids=embedding_id, include=['documents'])
            self.embeddings.append({
                'id': result['ids'][0],
                'work': result['documents'][0],
            })
            if index > 0 and index % 1000 == 0:
                print(f'Loaded {index}...')

    def search(self, work_embeddings, number_of_results=2):
        """
        Query Chroma for results near the individual XOS Embeddings JSON handed in.
        """
        return self.collection.query(
            query_embeddings=[work_embeddings],
            n_results=number_of_results,
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


with application.app_context():
    print('===================================')
    if not CHROMA:
        CHROMA = Chroma()
        CHROMA.get_collection()
        print(f'Loading embeddings from {DATABASE_PATH}...')
        CHROMA.load_embeddings()
        if REBUILD:
            print('Rebuilding the collection...')
            CHROMA.add_pages_of_embeddings()
    print('===================================')


if __name__ == '__main__':
    application.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG,
        use_reloader=False,
    )
