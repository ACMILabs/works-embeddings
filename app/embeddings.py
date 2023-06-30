import os
import random
import requests
from flask import Flask, render_template, request

import chromadb

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
XOS_API_ENDPOINT = os.getenv('XOS_API_ENDPOINT', None)
XOS_RETRIES = int(os.getenv('XOS_RETRIES', '3'))
XOS_TIMEOUT = int(os.getenv('XOS_TIMEOUT', '60'))

application = Flask(__name__)
embeddings = []
CHROMA = None


@application.route('/')
def home():
    """
    Works embeddings home page.
    """
    results = None
    filtered_embeddings = embeddings

    work = request.args.get('work', None)
    if work:
        for embedding in embeddings:
            if str(embedding['work']) == work:
                results = CHROMA.search(embedding, 11)
        if results:
            filtered_embeddings = []
            for index, embedding_id in enumerate(results['ids'][0]):
                for embedding in embeddings:
                    if str(embedding['id']) == embedding_id:
                        embedding['distance'] = results['distances'][0][index]
                        filtered_embeddings.append(embedding)
    elif filtered_embeddings:
        filtered_embeddings = random.sample(filtered_embeddings, 11)

    return render_template(
        'index.html',
        embeddings=filtered_embeddings,
    )


class NoCollectionException(Exception):
    """
    This exception is raised if a collection doesn't exist.
    """


class Chroma():
    """
    Chroma vector database to interact with embeddings.
    """
    def __init__(self):
        self.client = chromadb.Client()
        self.collection_name = 'works'
        self.collection = None

    def create_collection(self, name=None):
        """
        Create a collection of embeddings.
        """
        if not name:
            name = self.collection_name
        self.collection = self.client.create_collection(name=name)
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

    def add_pages_of_embeddings(self, pages=1):
        """
        Add pages of XOS Embeddings API results to Chroma.
        """
        page = 1
        while page < (pages + 1):
            embeddings_json = XOSAPI().get('embeddings', {'page': page}).json()
            for embedding in embeddings_json['results']:
                self.add_embedding(embedding)
                embeddings.append(embedding)
            print(f'Added {len(embeddings_json["results"])} from page {page}')
            page += 1

    def search(self, embeddings_json, number_of_results=2):
        """
        Query Chroma for results near the individual XOS Embeddings JSON handed in.
        """
        return self.collection.query(
            query_embeddings=[embeddings_json['data']['data'][0]['embedding']],
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


if __name__ == '__main__':
    if DEBUG:
        print('===================================')
        print('Adding some embeddings to Chroma...')
        CHROMA = Chroma()
        CHROMA.create_collection()
        CHROMA.add_pages_of_embeddings(2)
        print('===================================')
    application.run(
        host='0.0.0.0',
        port=8081,
        debug=DEBUG,
    )
