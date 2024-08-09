import json
import os
import random
import time
import subprocess
import requests
from flask import Flask, jsonify, render_template, request, Response
from PIL import Image

import chromadb
import open_clip
import torch

DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
REBUILD = os.getenv('REBUILD', 'false').lower() == 'true'
PAGES = os.getenv('PAGES', '999999')
XOS_API_ENDPOINT = os.getenv('XOS_API_ENDPOINT', None)
XOS_RETRIES = int(os.getenv('XOS_RETRIES', '1'))
XOS_TIMEOUT = int(os.getenv('XOS_TIMEOUT', '120'))
DATABASE_PATH = os.getenv('DATABASE_PATH', '')
PORT = int(os.getenv('PORT', '8081'))
DEFAULT_TEMPLATE_JSON = os.getenv('DEFAULT_TEMPLATE_JSON', 'true').lower()
REFRESH_TIMEOUT = int(os.getenv('REFRESH_TIMEOUT', '0') or '0')
TEXT_SEARCH = os.getenv('TEXT_SEARCH', 'false').lower() == 'true'

LENS_READER_TAPS_API = os.getenv('LENS_READER_TAPS_API', None)
AUTH_TOKEN = os.getenv('AUTH_TOKEN', None)

application = Flask(__name__)
CHROMA = None
OPENCLIP = None
SELECTED_WORK_ID = None
SUCCESSFUL_TAP = None


def normalise_distance(distance, min_distance=30, max_distance=1200):
    """
    Returns a normalised distance between 0 (closest) and 1 (furthest).
    """
    return (distance - min_distance) / (max_distance - min_distance)


@application.template_filter('format_distance')
def format_distance(value):
    """
    Format the vector distance between embeddings to a percentage.
    """
    if value > 1:
        value = normalise_distance(value)
    return round((1 - value) * 100)


@application.template_filter('format_timestamp')
def format_timestamp(value):
    """
    Format the timestamp in minutes and seconds.
    """
    timestamp = value.split('_')[-1]
    minutes, seconds = divmod(float(timestamp), 60)
    return f'{int(minutes)}:{str(int(seconds)).zfill(2)}'


@application.route('/')
def home():
    """
    Works embeddings home page.
    """
    results = None
    json_response = request.args.get('json', DEFAULT_TEMPLATE_JSON).lower() == 'true'
    work = request.args.get('work', None)
    size = int(request.args.get('size', '11'))
    global SELECTED_WORK_ID  # pylint: disable=global-statement
    embeddings = CHROMA.embeddings.get('works')
    filtered_embeddings = embeddings

    if work:
        SELECTED_WORK_ID = work
        for embedding in embeddings:
            if str(embedding['work']) == work:
                result = CHROMA.collections['works'].get(
                    ids=embedding['id'],
                    include=['embeddings', 'documents'],
                )
                results = CHROMA.search(result['embeddings'][0], size)
        if results:
            filtered_embeddings = []
            for index, embedding_id in enumerate(results['ids'][0]):
                for embedding in embeddings:
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


@application.route('/videos/')
def videos():
    """
    Video embeddings page.
    """
    return video_and_image_response(request)


@application.route('/images/')
def images():
    """
    Video embeddings page.
    """
    return video_and_image_response(request)


def video_and_image_response(page_request):  # pylint: disable=too-many-branches
    """
    Generate a response from requests to the /videos/ or /images/ page.
    """
    results = None
    json_response = page_request.args.get('json', DEFAULT_TEMPLATE_JSON).lower() == 'true'
    text = page_request.args.get('text', None)
    image = page_request.args.get('image', None)
    work = page_request.args.get('work', None)
    size = int(page_request.args.get('size', '11'))
    path = page_request.path.replace('/', '')
    global SELECTED_WORK_ID  # pylint: disable=global-statement
    embeddings = CHROMA.embeddings.get(path)
    filtered_embeddings = embeddings

    if (image or text) and TEXT_SEARCH:
        SELECTED_WORK_ID = None
        # Create OpenCLIP embedding of the query
        clip = OPENCLIP.get_embeddings(image=image, text_string=text, openai_format=False)[0]
        results = CHROMA.search(clip, size, collection_name=path)
        if results:
            filtered_embeddings = []
            for index, embedding_id in enumerate(results['ids'][0]):
                for embedding in embeddings:
                    if str(embedding['id']) == embedding_id:
                        embedding['distance'] = results['distances'][0][index]
                        filtered_embeddings.append(embedding)
    elif work:  # pylint: disable=too-many-nested-blocks
        SELECTED_WORK_ID = work_id_from_item_id(work)
        for embedding in embeddings:
            if str(embedding['work']) == work:
                result = CHROMA.collections[path].get(
                    ids=embedding['id'],
                    include=['embeddings', 'documents'],
                )
                results = CHROMA.search(result['embeddings'][0], size, collection_name=path)
                if results:
                    filtered_embeddings = []
                    for index, embedding_id in enumerate(results['ids'][0]):
                        for embedding in embeddings:
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
            text=text,
            path=path,
            text_search=TEXT_SEARCH,
        )

    return jsonify(filtered_embeddings)


@application.route('/thumbnail/<item_id>/')
def thumbnail(item_id):
    """
    Returns a thumbnail image from an image ID or video ID at a given timestamp.
    """
    image_path = None
    max_image_dimension = 600
    item_id_only = item_id.split('_')[1]
    timestamp = item_id.split('_')[-1]
    tmp_image_path = os.path.join(application.root_path, 'static', 'cache', f'{item_id}.jpg')
    if os.path.isfile(tmp_image_path):
        image_path = os.path.join('static', 'cache', f'{item_id}.jpg')
    elif 'image' in item_id:
        xos = XOSAPI()
        image_json = xos.get(f'images/{item_id_only}').json()
        image_url = image_json.get('image_file_thumbnail')
        img_data = requests.get(image_url, timeout=10).content
        os.makedirs(os.path.dirname(tmp_image_path), exist_ok=True)
        with open(tmp_image_path, 'wb') as image_file:
            image_file.write(img_data)
        image_path = os.path.join('static', 'cache', f'{item_id}.jpg')
    else:
        xos = XOSAPI()
        video_json = xos.get(f'assets/{item_id_only}').json()
        video_url = video_json.get('web_resource') or video_json.get('resource')
        if video_url:
            command = [
                'ffmpeg',
                '-ss', str(timestamp),
                '-i', video_url,
                '-vframes', '1',
                '-q:v', '10',
                '-vf', f"scale='if(gt(iw,{max_image_dimension}),{max_image_dimension},iw)':"
                       f"'if(gt(ih,{max_image_dimension}),{max_image_dimension},ih)'",
                tmp_image_path,
            ]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            image_path = os.path.join('static', 'cache', f'{item_id}.jpg')
    return jsonify({'thumbnail': image_path})


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
        work_id = work_id_from_item_id(work_id)
        if work_id:
            return jsonify(xos.get(f'works/{work_id}').json())
        return jsonify({'error': 'No work attached to this video.'}), 404


def work_id_from_item_id(item_id):
    """
    Returns the Work ID from a Video or Image ID.
    """
    work_id = None
    item_id_parts = item_id.split('_')
    if len(item_id_parts) > 1:
        if item_id_parts[0] == 'video' or item_id_parts[0] == 'image':
            try:
                work_id = int(item_id_parts[2])
            except (IndexError, ValueError):
                pass
        elif item_id_parts[0] == 'work':
            work_id = item_id_parts[1]
    else:
        work_id = item_id
    return work_id


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
        self.collections = {'works': None, 'videos': None, 'images': None}
        self.embeddings = {'works': [], 'videos': [], 'images': []}

    def get_collection(self, name='works'):
        """
        Get or create a collection of embeddings.
        """
        self.collections[name] = self.client.get_or_create_collection(name=name)
        return self.collections[name]

    def add_embedding(self, embeddings_json, collection_name='works'):
        """
        Add an embedding to Chroma from an individual XOS Embeddings JSON.
        """
        embeddings_added = []
        if not self.collections.get(collection_name):
            raise NoCollectionException(f'Please create or load the collection {collection_name}')
        for embedding_item in embeddings_json['data']['data']:
            prefix = 'work_'
            suffix = ''
            item_id = f"work_{embeddings_json['work']}"
            if embeddings_json['video']:
                prefix = 'video_'
                item_id = f"{embeddings_json['video']['id']}_{embeddings_json['video']['work_id']}"
                if embedding_item.get('timestamp') == 0.0:
                    continue
            if embeddings_json['image']:
                prefix = 'image_'
                item_id = f"{embeddings_json['image']['id']}_{embeddings_json['image']['work_id']}"
                if not embeddings_json['image'].get('work_id'):
                    continue
            if embedding_item.get('timestamp') is not None:
                suffix = f"_{embedding_item['timestamp']}"
            embeddings_added.append(
                self.collections[collection_name].add(
                    embeddings=[embedding_item['embedding']],
                    documents=[f"{prefix}{item_id}{suffix}"],
                    metadatas=[{'source': 'embeddings'}],
                    ids=[f"{prefix}{embeddings_json['id']}{suffix}"],
                )
            )
        return embeddings_added

    def add_pages_of_embeddings(self, pages=int(PAGES), embedding_type='works'):
        """
        Add pages of XOS Embeddings API results to Chroma.
        """
        page = 1
        while page < (pages + 1):
            params = {'page': page, 'page_size': 10, 'ordering': 'id'}
            if embedding_type == 'works':
                params['only_works'] = 'true'
                params['unpublished'] = 'false'
            elif embedding_type == 'videos':
                params['only_videos'] = 'true'
                params['unpublished'] = 'false'
            elif embedding_type == 'images':
                params['only_images'] = 'true'
                params['unpublished'] = 'false'
                params['page_size'] = 100
            embeddings_json = XOSAPI().get(
                'embeddings',
                params,
            ).json()
            for embedding in embeddings_json['results']:
                self.add_embedding(embedding, collection_name=embedding_type)
            print(f'Added {len(embeddings_json["results"])} {embedding_type} page {page}')
            page += 1
            if not embeddings_json['next']:
                print(
                    f'Finished adding {len(self.embeddings.get(embedding_type))} {embedding_type}.',
                )
                break

    def load_embeddings(self, collection_name='works'):
        """
        Load embeddings from the collection.
        """
        if not self.collections.get(collection_name):
            self.get_collection(name=collection_name)
        embedding_ids = self.collections[collection_name].get()['ids']
        for index, embedding_id in enumerate(embedding_ids):
            result = self.collections[collection_name].get(ids=embedding_id, include=['documents'])
            self.embeddings[collection_name].append({
                'id': result['ids'][0],
                'work': result['documents'][0],
            })
            if index > 0 and index % 1000 == 0:
                print(f'Loaded {index}...')

    def search(self, work_embeddings, number_of_results=2, collection_name='works'):
        """
        Query Chroma for results near the individual XOS Embeddings JSON handed in.
        """
        if not self.collections.get(collection_name):
            raise NoCollectionException(f'Please create or load the collection {collection_name}')
        return self.collections[collection_name].query(
            query_embeddings=[work_embeddings],
            n_results=number_of_results,
        )


class ImageEmbedding():
    """
    Create image embeddings using OpenCLIP.
    """
    def __init__(self):
        self.model_name = 'ViT-g-14'
        self.pretrained = 'laion2b_s34b_b88k'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.sample_frequency = 100

    def get_embeddings(self, image=None, image_path=None, text_string=None, openai_format=True):
        """
        Generate embeddings for an image or text string.
        """
        embeddings = None
        preprocessed_image = None
        tokens = 0

        if image_path or image:
            if image_path:
                preprocessed_image = self.preprocess(Image.open(image_path)).unsqueeze(0)
            elif image:
                preprocessed_image = self.preprocess(image).unsqueeze(0)
            tokens = self.model.visual.positional_embedding.shape[0] - 1
            if preprocessed_image:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    tensors = self.model.encode_image(preprocessed_image)
                    embeddings = tensors.cpu().numpy().tolist()[0]

        if text_string:
            preprocessed_text = self.tokenizer(text_string)
            tokens = preprocessed_text.shape[-1]
            with torch.no_grad(), torch.cuda.amp.autocast():
                tensors = self.model.encode_text(preprocessed_text)
                embeddings = tensors.cpu().numpy().tolist()[0]

        if embeddings and openai_format:
            embeddings = self.openai_embeddings_format(embeddings, tokens)

        return embeddings, tokens

    def openai_embeddings_format(self, embeddings, tokens):
        """
        Returns OpenAI Embeddings format from a list of embeddings.
        """
        return {
            'data': [{
                'index': 0,
                'object': 'embedding',
                'embedding': embeddings,
            }],
            'model': f'{self.model_name} {self.pretrained}',
            'usage': {
                'total_tokens': tokens,
                'prompt_tokens': tokens,
            },
            'object': 'list',
        }


class XOSAPI():  # pylint: disable=too-few-public-methods
    """
    XOS private API interface.
    """
    def __init__(self):
        self.uri = XOS_API_ENDPOINT
        self.headers = {
            'Content-Type': 'application/json',
        }
        if AUTH_TOKEN:
            self.headers['Authorization'] = f'Token {AUTH_TOKEN}'
        self.params = {
            'page_size': 10,
            'ordering': 'id',
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
                time.sleep(5)
                if retries == XOS_RETRIES:
                    raise exception
        return None


with application.app_context():
    print('===================================')
    if not OPENCLIP and TEXT_SEARCH:
        print('Starting up OpenCLIP...')
        OPENCLIP = ImageEmbedding()
    if not CHROMA:
        CHROMA = Chroma()
        CHROMA.get_collection()
        if REBUILD:
            print('Rebuilding the collection...')
            CHROMA.add_pages_of_embeddings()
            print('Finished works...')
            CHROMA.get_collection('videos')
            CHROMA.add_pages_of_embeddings(embedding_type='videos')
            print('Finished videos...')
            CHROMA.get_collection('images')
            CHROMA.add_pages_of_embeddings(embedding_type='images')
            print('Finished images...')
        print(f'Loading embeddings from {DATABASE_PATH}...')
        CHROMA.load_embeddings()
        CHROMA.get_collection('videos')
        CHROMA.load_embeddings('videos')
        CHROMA.get_collection('images')
        CHROMA.load_embeddings('images')
    print('===================================')


if __name__ == '__main__':
    application.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG,
        use_reloader=False,
        threaded=True,
    )
