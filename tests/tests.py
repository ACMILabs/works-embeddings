from unittest.mock import patch
from app.embeddings import application


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
