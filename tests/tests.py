from unittest.mock import patch
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
        assert 'ACMI Collections explorer' in response.text


@patch('app.embeddings.chromadb')
@patch('app.embeddings.open_clip')
def test_images(_, __):
    """
    Test the Collections images embeddings returns expected content.
    """
    with application.test_client() as client:
        response = client.get('/images/?json=false')
        assert response.status_code == 200
        assert 'ACMI Collections images explorer' in response.text


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
