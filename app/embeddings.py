import os
import random
import requests
from flask import Flask, jsonify, render_template, request

import chromadb

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
REBUILD = os.getenv('REBUILD', 'false').lower() == 'true'
XOS_API_ENDPOINT = os.getenv('XOS_API_ENDPOINT', None)
XOS_RETRIES = int(os.getenv('XOS_RETRIES', '1'))
XOS_TIMEOUT = int(os.getenv('XOS_TIMEOUT', '60'))
DATABASE_PATH = os.getenv('DATABASE_PATH', '')

application = Flask(__name__)
CHROMA = None


@application.route('/')
def home():
    """
    Works embeddings home page.
    """
    results = None
    filtered_embeddings = CHROMA.embeddings

    json = request.args.get('json', 'false').lower() == 'true'
    work = request.args.get('work', None)
    size = int(request.args.get('size', '11'))

    if work:
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

    if DEBUG and not json:
        return render_template(
            'index.html',
            embeddings=filtered_embeddings,
        )

    return jsonify(filtered_embeddings)


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
        print('Loading embeddings...')
        CHROMA.load_embeddings()
        if REBUILD:
            print('Rebuilding the collection...')
            CHROMA.add_pages_of_embeddings()
    print('===================================')


if __name__ == '__main__':
    application.run(
        host='0.0.0.0',
        port=8081,
        debug=DEBUG,
        use_reloader=False,
    )
