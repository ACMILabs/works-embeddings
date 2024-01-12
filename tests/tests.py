from app.embeddings import application


def test_root():
    """
    Test the Works embeddings root returns expected content.
    """
    with application.test_client() as client:
        response = client.get('/?json=false')
        assert response.status_code == 200
        assert 'ACMI works explorer' in response.text
