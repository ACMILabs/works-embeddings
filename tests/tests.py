from unittest.mock import patch

import app.embeddings as embeddings_module
from app.embeddings import application, format_distance, format_timestamp, normalise_distance


@patch('app.embeddings.chromadb')
@patch('app.embeddings.open_clip')
def test_root(_, __):
    """
    Test the Collections embeddings root returns expected content.
    """
    with application.test_client() as client:
        response = client.get('/?json=false')
        assert response.status_code == 200
        assert 'ACMI Collection explorer' in response.text
        assert 'Empty vector database' in response.text


@patch('app.embeddings.chromadb')
@patch('app.embeddings.open_clip')
def test_images(_, __):
    """
    Test the Collections images embeddings returns expected content.
    """
    with application.test_client() as client:
        response = client.get('/images/?json=false')
        assert response.status_code == 200
        assert 'ACMI Collection images explorer' in response.text


@patch('app.embeddings.chromadb')
@patch('app.embeddings.open_clip')
def test_videos(_, __):
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
