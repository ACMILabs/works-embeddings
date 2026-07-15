from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import app.embeddings as embeddings_module
from app.embeddings import (
    application,
    count_clip_input_tokens,
    format_distance,
    format_timestamp,
    get_model_device,
    get_pipeline_device_kwargs,
    get_projected_features,
    get_tensor_shape,
    normalise_distance,
    tensor_to_embedding,
)


class FakeCollection:
    """
    Minimal Chroma collection double for loader tests.
    """
    def __init__(self, items):
        self.items = items
        self.get_calls = []

    def count(self):
        """
        Return the fake collection size.
        """
        return len(self.items)

    def get(self, limit=None, offset=None, include=None):
        """
        Return one page of fake collection data.
        """
        self.get_calls.append({
            'limit': limit,
            'offset': offset,
            'include': include,
        })
        page = self.items[offset:offset + limit]
        return {
            'ids': [item['id'] for item in page],
            'documents': [item['document'] for item in page],
        }


class FakeTensor:
    """
    Minimal tensor double for CLIP helper tests.
    """
    def __init__(self, value=None, shape=None):
        self.value = value
        self.shape = shape or []

    def sum(self):
        """
        Return this tensor for chained sum().item() calls.
        """
        return self

    def item(self):
        """
        Return the scalar test value.
        """
        return self.value

    def detach(self):
        """
        Return this tensor for chained detach().cpu().flatten().tolist() calls.
        """
        return self

    def cpu(self):
        """
        Return this tensor for chained cpu().flatten().tolist() calls.
        """
        return self

    def flatten(self):
        """
        Return this tensor for chained flatten().tolist() calls.
        """
        return self

    def tolist(self):
        """
        Return the list test value.
        """
        return self.value


class FakeInputs(dict):
    """
    Processor input double that records device moves.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None

    def to(self, device):  # pylint: disable=invalid-name
        """
        Record the device that inputs were moved to.
        """
        self.device = device
        return self


@patch('app.embeddings.chromadb')
@patch('app.embeddings.AutoProcessor')
@patch('app.embeddings.CLIPModel')
def test_root(_, __, ___):
    """
    Test the Collections embeddings root returns expected content.
    """
    with patch.object(embeddings_module, 'CHROMA', None):
        with application.test_client() as client:
            response = client.get('/?json=false')
            assert response.status_code == 200
            assert 'ACMI Collection explorer' in response.text
            assert 'Empty vector database' in response.text


@patch('app.embeddings.chromadb')
@patch('app.embeddings.AutoProcessor')
@patch('app.embeddings.CLIPModel')
def test_images(_, __, ___):
    """
    Test the Collections images embeddings returns expected content.
    """
    with application.test_client() as client:
        response = client.get('/images/?json=false')
        assert response.status_code == 200
        assert 'ACMI Collection images explorer' in response.text


@patch('app.embeddings.chromadb')
@patch('app.embeddings.AutoProcessor')
@patch('app.embeddings.CLIPModel')
def test_videos(_, __, ___):
    """
    Test the Collections videos embeddings returns expected content.
    """
    with application.test_client() as client:
        response = client.get('/videos/?json=false')
        assert response.status_code == 200
        assert 'ACMI Collection videos explorer' in response.text


def test_normalise_distance():
    """
    Test normalising a distance value functions as expected.
    """
    assert normalise_distance(40) == 0.008547008547008548
    assert normalise_distance(30) == 0.0
    assert normalise_distance(615) == 0.5
    assert normalise_distance(1200) == 1.0


def test_format_distance():
    """
    Test foramtting the distance as a percentage.
    """
    assert format_distance(0.2) == 80
    assert format_distance(0.8) == 20
    assert format_distance(615) == 50


def test_format_timestamp():
    """
    Test foramtting the video_id to a timestamp.
    """
    assert format_timestamp('123_45.0') == '0:45'
    assert format_timestamp('123_189.0') == '3:09'
    assert format_timestamp('123_999.123') == '16:39'


def test_get_pipeline_device_kwargs():
    """
    Device kwargs are parsed from the TRANSFORMERS_DEVICE environment variable.
    """
    with patch.dict('app.embeddings.os.environ', {}, clear=True):
        assert not get_pipeline_device_kwargs()
    with patch.dict('app.embeddings.os.environ', {'TRANSFORMERS_DEVICE': '0'}):
        assert get_pipeline_device_kwargs() == {'device': 0}
    with patch.dict('app.embeddings.os.environ', {'TRANSFORMERS_DEVICE': 'mps'}):
        assert get_pipeline_device_kwargs() == {'device': 'mps'}


def test_get_model_device():
    """
    Pipeline devices are converted to torch model devices.
    """
    with patch('app.embeddings.get_pipeline_device_kwargs', return_value={}):
        assert get_model_device() is None
    with patch('app.embeddings.get_pipeline_device_kwargs', return_value={'device': -1}):
        assert get_model_device() is None
    with patch('app.embeddings.get_pipeline_device_kwargs', return_value={'device': 0}):
        assert get_model_device() == 'cuda:0'
    with patch('app.embeddings.get_pipeline_device_kwargs', return_value={'device': 'mps'}):
        assert get_model_device() == 'mps'


def test_get_tensor_shape():
    """
    Tensor-like shapes and nested lists are handled.
    """
    assert get_tensor_shape(FakeTensor(shape=[1, 3, 336, 336])) == [1, 3, 336, 336]
    assert get_tensor_shape([[1, 2, 3]]) == [1, 3]
    assert get_tensor_shape([]) == [0]


def test_count_clip_input_tokens_for_images_and_text():
    """
    CLIP image and text token counts are inferred from processor inputs.
    """
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=14),
        ),
    )
    assert count_clip_input_tokens(
        {'pixel_values': FakeTensor(shape=[1, 3, 336, 336])},
        model,
        is_image=True,
    ) == 577
    assert count_clip_input_tokens({}, model, is_image=True) == 0
    assert count_clip_input_tokens({'attention_mask': FakeTensor(value=4)}, model) == 4
    assert count_clip_input_tokens({'input_ids': [[1, 2, 3]]}, model) == 3
    assert count_clip_input_tokens({}, model) == 0


def test_tensor_to_embedding_extracts_projected_features():
    """
    Projected CLIP outputs are flattened to one embedding vector.
    """
    assert tensor_to_embedding({'image_embeds': [[1, 2], [3, 4]]}) == [1.0, 2.0, 3.0, 4.0]
    assert tensor_to_embedding(SimpleNamespace(pooler_output=FakeTensor(value=[[5, 6]]))) == [5.0, 6.0]


def test_get_projected_features_rejects_unprojected_outputs():
    """
    Dict-like CLIP outputs must include projected embedding features.
    """
    with pytest.raises(ValueError):
        get_projected_features({'last_hidden_state': [[1, 2]]})


@patch('app.embeddings.get_model_device', return_value='mps')
@patch('app.embeddings.CLIPModel')
@patch('app.embeddings.AutoProcessor')
def test_image_embedding_initialises_transformers(mock_processor, mock_model, _):
    """
    ImageEmbedding loads the Transformers CLIP processor and model.
    """
    parameter = MagicMock()
    model = MagicMock()
    model.parameters.return_value = [parameter]
    mock_model.from_pretrained.return_value = model

    image_embedding = embeddings_module.ImageEmbedding()

    mock_processor.from_pretrained.assert_called_once_with('openai/clip-vit-large-patch14-336')
    mock_model.from_pretrained.assert_called_once_with('openai/clip-vit-large-patch14-336')
    model.to.assert_called_once_with('mps')
    parameter.requires_grad_.assert_called_once_with(False)
    model.eval.assert_called_once_with()
    assert image_embedding.model_name == 'openai/clip-vit-large-patch14-336'


@patch('app.embeddings.get_model_device', return_value='mps')
@patch('app.embeddings.CLIPModel')
@patch('app.embeddings.AutoProcessor')
def test_image_embedding_get_text_embeddings(mock_processor, mock_model, _):
    """
    Text query embeddings are generated through Transformers CLIP text features.
    """
    inputs = FakeInputs({'attention_mask': FakeTensor(value=3)})
    processor = MagicMock(return_value=inputs)
    mock_processor.from_pretrained.return_value = processor
    model = MagicMock()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(max_position_embeddings=77),
    )
    model.parameters.return_value = []
    model.get_text_features.return_value = [[0.1, 0.2]]
    mock_model.from_pretrained.return_value = model

    image_embedding = embeddings_module.ImageEmbedding()
    embeddings, tokens = image_embedding.get_embeddings(text_string='hello', openai_format=False)

    processor.assert_called_with(
        text='hello',
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=77,
    )
    assert inputs.device == 'mps'
    model.get_text_features.assert_called_once_with(**inputs)
    assert embeddings == [0.1, 0.2]
    assert tokens == 3


@patch('app.embeddings.get_model_device', return_value=None)
@patch('app.embeddings.CLIPModel')
@patch('app.embeddings.AutoProcessor')
def test_image_embedding_get_image_embeddings(mock_processor, mock_model, _):
    """
    Image query embeddings are generated through Transformers CLIP image features.
    """
    image = object()
    inputs = FakeInputs({'pixel_values': FakeTensor(shape=[1, 3, 336, 336])})
    processor = MagicMock(return_value=inputs)
    mock_processor.from_pretrained.return_value = processor
    model = MagicMock()
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=14),
    )
    model.parameters.return_value = []
    model.get_image_features.return_value = [[0.3, 0.4]]
    mock_model.from_pretrained.return_value = model

    image_embedding = embeddings_module.ImageEmbedding()
    embeddings, tokens = image_embedding.get_embeddings(image=image, openai_format=False)

    processor.assert_called_with(images=image, return_tensors='pt', padding=True)
    model.get_image_features.assert_called_once_with(**inputs)
    assert embeddings == [0.3, 0.4]
    assert tokens == 577


@patch('app.embeddings.Image.open')
@patch('app.embeddings.os.path.exists', return_value=True)
@patch('app.embeddings.get_model_device', return_value=None)
@patch('app.embeddings.CLIPModel')
@patch('app.embeddings.AutoProcessor')
def test_image_embedding_accepts_local_image_paths(
        mock_processor, mock_model, _, mock_exists, mock_image_open):
    """
    String image queries must point to local files inside the container.
    """
    image = object()
    opened_image = MagicMock()
    opened_image.__enter__.return_value.copy.return_value = image
    mock_image_open.return_value = opened_image
    inputs = FakeInputs({'pixel_values': FakeTensor(shape=[1, 3, 336, 336])})
    processor = MagicMock(return_value=inputs)
    mock_processor.from_pretrained.return_value = processor
    model = MagicMock()
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(patch_size=14),
    )
    model.parameters.return_value = []
    model.get_image_features.return_value = [[0.3, 0.4]]
    mock_model.from_pretrained.return_value = model

    image_embedding = embeddings_module.ImageEmbedding()
    image_embedding.get_embeddings(image='/code/query-images/image.jpg', openai_format=False)

    mock_exists.assert_called_once_with('/code/query-images/image.jpg')
    mock_image_open.assert_called_once_with('/code/query-images/image.jpg')
    processor.assert_called_with(images=image, return_tensors='pt', padding=True)


@patch('app.embeddings.get_model_device', return_value=None)
@patch('app.embeddings.CLIPModel')
@patch('app.embeddings.AutoProcessor')
def test_image_embedding_rejects_non_local_image_strings(mock_processor, mock_model, _):
    """
    String image queries are rejected when the path is not local.
    """
    mock_processor.from_pretrained.return_value = MagicMock()
    model = MagicMock()
    model.parameters.return_value = []
    mock_model.from_pretrained.return_value = model

    image_embedding = embeddings_module.ImageEmbedding()

    with pytest.raises(ValueError):
        image_embedding.get_embeddings(image='https://example.com/image.jpg', openai_format=False)


@patch('app.embeddings.CHROMA_LOAD_PAGE_SIZE', 2)
@patch('app.embeddings.chromadb')
def test_chroma_load_embeddings_pages_collection(_):
    """
    Chroma embeddings are loaded in bounded pages.
    """
    fake_collection = FakeCollection([
        {'id': '1', 'document': 'work_1'},
        {'id': '2', 'document': 'work_2'},
        {'id': '3', 'document': 'work_3'},
    ])
    chroma = embeddings_module.Chroma()
    chroma.collections['works'] = fake_collection

    chroma.load_embeddings()

    assert chroma.embeddings['works'] == [
        {'id': '1', 'work': 'work_1'},
        {'id': '2', 'work': 'work_2'},
        {'id': '3', 'work': 'work_3'},
    ]
    assert fake_collection.get_calls == [
        {'limit': 2, 'offset': 0, 'include': ['documents']},
        {'limit': 2, 'offset': 2, 'include': ['documents']},
    ]


@patch('app.embeddings.chromadb')
def test_chroma_load_embeddings_replaces_existing_cache(_):
    """
    Reloading a collection replaces the in-memory embedding cache.
    """
    fake_collection = FakeCollection([
        {'id': '1', 'document': 'work_1'},
    ])
    chroma = embeddings_module.Chroma()
    chroma.collections['works'] = fake_collection
    chroma.embeddings['works'] = [{'id': 'stale', 'work': 'stale_work'}]

    chroma.load_embeddings()

    assert chroma.embeddings['works'] == [{'id': '1', 'work': 'work_1'}]


def test_load_status_not_started():
    """
    Status endpoint reports not_started before loading starts.
    """
    with patch.object(embeddings_module, 'LOADED', False), \
            patch.object(embeddings_module, 'LOADING', False), \
            patch.object(embeddings_module, 'LOAD_ERROR', None):
        with application.test_client() as client:
            response = client.get('/load?status=true')
            assert response.status_code == 200
            assert response.json == {'status': 'not_started'}


def test_load_status_loading():
    """
    Status endpoint reports loading while a load is in progress.
    """
    with patch.object(embeddings_module, 'LOADED', False), \
            patch.object(embeddings_module, 'LOADING', True), \
            patch.object(embeddings_module, 'LOAD_ERROR', None):
        with application.test_client() as client:
            response = client.get('/load?status=true')
            assert response.status_code == 200
            assert response.json == {'status': 'loading'}


def test_load_status_loaded():
    """
    Status endpoint reports loaded once loading is complete.
    """
    with patch.object(embeddings_module, 'LOADED', True), \
            patch.object(embeddings_module, 'LOADING', False), \
            patch.object(embeddings_module, 'LOAD_ERROR', None):
        with application.test_client() as client:
            response = client.get('/load?status=true')
            assert response.status_code == 200
            assert response.json == {'status': 'loaded'}


def test_load_status_error():
    """
    Status endpoint reports loader errors.
    """
    with patch.object(embeddings_module, 'LOADED', False), \
            patch.object(embeddings_module, 'LOADING', False), \
            patch.object(embeddings_module, 'LOAD_ERROR', 'boom'):
        with application.test_client() as client:
            response = client.get('/load?status=true')
            assert response.status_code == 200
            assert response.json == {'status': 'error', 'error': 'boom'}


@patch('app.embeddings.threading.Thread')
def test_load_starts_background_thread(mock_thread):
    """
    GET /load triggers loading in a background thread.
    """
    thread = mock_thread.return_value
    with patch.object(embeddings_module, 'LOADED', False), \
            patch.object(embeddings_module, 'LOADING', False), \
            patch.object(embeddings_module, 'LOAD_ERROR', 'previous_error'):
        with application.test_client() as client:
            response = client.get('/load')
            assert response.status_code == 202
            assert response.json == {'status': 'loading'}
        assert embeddings_module.LOADING is True
        assert embeddings_module.LOAD_ERROR is None

    mock_thread.assert_called_once_with(
        target=embeddings_module.run_loader_in_background,
        daemon=True,
    )
    thread.start.assert_called_once_with()


@patch('app.embeddings.threading.Thread')
def test_load_returns_loaded_when_ready(mock_thread):
    """
    /load returns loaded when loader has already completed.
    """
    with patch.object(embeddings_module, 'LOADED', True), \
            patch.object(embeddings_module, 'LOADING', False):
        with application.test_client() as client:
            response = client.get('/load')
            assert response.status_code == 200
            assert response.json == {'status': 'loaded'}
    mock_thread.assert_not_called()


@patch('app.embeddings.threading.Thread')
def test_load_returns_loading_when_in_progress(mock_thread):
    """
    /load returns loading when loader is already in progress.
    """
    with patch.object(embeddings_module, 'LOADED', False), \
            patch.object(embeddings_module, 'LOADING', True):
        with application.test_client() as client:
            response = client.get('/load')
            assert response.status_code == 202
            assert response.json == {'status': 'loading'}
    mock_thread.assert_not_called()


@patch('app.embeddings.run_loader', side_effect=Exception('boom'))
def test_run_loader_in_background_sets_error_and_resets_loading(_):
    """
    Background loader wrapper persists error and clears loading state.
    """
    with patch.object(embeddings_module, 'LOADING', True), \
            patch.object(embeddings_module, 'LOAD_ERROR', None):
        embeddings_module.run_loader_in_background()
        assert embeddings_module.LOADING is False
        assert embeddings_module.LOAD_ERROR == 'boom'
