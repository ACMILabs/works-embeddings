# ACMI Works embeddings

A recommendation engine for Works in the ACMI Collection using [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings).

![ACMI Works Embeddings CI](https://github.com/ACMILabs/works-embeddings/workflows/ACMI%20Works%20Embeddings%20CI/badge.svg)

<img src="works-explorer-home-v2.png" width="50%" alt="ACMI Works explorer frontend" title="ACMI Works explorer frontend" /><img src="works-explorer-home-json-v2.png" width="50%" alt="ACMI Works explorer JSON server" title="ACMI Works explorer JSON server" />

## Use

* Connect to your ACMI VPN to access XOS private APIs (or point to your own API)
* Copy the `config.tmpl.env` file to `config.env`
* Set `DEFAULT_TEMPLATE_JSON=false` if you'd like to see HTML results rather than JSON results
* Start your environment: `make up`
* Click a work to get its nearest neighbours: http://localhost:8081/?json=false

### Text and image search

The `/images/` and `/videos/` explorers can search by text or image query when `TEXT_SEARCH=true` is set in `config.env`.

```text
TEXT_SEARCH=true
```

Text queries use Hugging Face Transformers and `openai/clip-vit-large-patch14-336` to create a CLIP embedding:

* http://localhost:8081/images/?json=false&text=red%20dress
* http://localhost:8081/videos/?json=false&text=city%20street

Image queries can pass a local image path inside the container or a URL:

* http://localhost:8081/images/?json=false&image=https://example.com/image.jpg

The first startup with `TEXT_SEARCH=true` downloads the CLIP model from Hugging Face. Set `HF_TOKEN` for higher Hugging Face Hub rate limits if needed. By default the model runs on CPU; set `TRANSFORMERS_DEVICE` to a supported PyTorch device such as `0` for `cuda:0` or `mps` for Apple Silicon.

Chroma collections must contain embeddings with the same dimensions as the query model. `openai/clip-vit-large-patch14-336` returns 768-dimensional vectors, so rebuild the image/video collections after switching from the old OpenCLIP model.

### Rebuild the Chroma database from your Embeddings API

* Open `config.env` and set `REBUILD=true`
* Delete the `works_db` directory if it exists, especially after changing embedding models
* Start the app: `make up`

## Create Embeddings

This prototype relies on having already created OpenAI Embeddings for your collection database.

Code we use to create Embeddings:

```python
def create_embeddings(self, work):
    """
    Create an Embedding from a Work.
    """
    embedding = None
    work_features = [
        work.get_title_display(),
        work.description_override or work.description,
        work.work_type,
        work.creator_credit(),
        work.headline_credit(),
    ]
    work_features = list(filter(None, work_features))
    text_string = '\n'.join(work_features)
    embeddings_json = self.get_embeddings(text_string)
    if embeddings_json:
        embedding, _ = Embedding.objects.get_or_create(
            work=work,
            defaults={'data': embeddings_json},
        )
        embedding.data = embeddings_json
        embedding.save()
    return embedding
```

An example of the resulting `JSON` Embedding model from the XOS `/embeddings/` API endpoint:

```json
{
  "id": 6826,
  "data": {
    "data": [
      {
        "index": 0,
        "object": "embedding",
        "embedding": [
          0.010930221527814865, -0.01788223721086979, 0.009138058871030807,
          -0.0015344980638474226, 0.00028023053891956806, 0.015440168790519238,
          ...
        ]
      }
    ],
    "model": "text-embedding-ada-002-v2",
    "usage": { "total_tokens": 101, "prompt_tokens": 101 },
    "object": "list"
  },
  "work": 108230
}
```

## TODO

- [x] Submodule Chroma vector database
- [x] Build Flask interface for prototyping
- [x] Load Chroma with XOS Works embeddings
- [x] Get recommendations based on an ACMI collection Work
- [x] Remove Chroma submodule if not necessary
- [x] Fix CORS issue loading images locally
