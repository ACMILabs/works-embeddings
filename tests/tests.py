from app.embeddings import application


def test_root():
    """
    Test the Collections embeddings root returns expected content.
    """
    with application.test_client() as client:
        response = client.get('/?json=false')
        assert response.status_code == 200
        assert 'ACMI Collections explorer' in response.text
