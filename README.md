# ACMI Works embeddings

A recommendation engine for Works in the ACMI Collection using [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings).

![ACMI Works Embeddings CI](https://github.com/ACMILabs/works-embeddings/workflows/ACMI%20Works%20Embeddings%20CI/badge.svg)

## Use

* Start your environment: `make base`
* Click a work to get its nearest neighbours: http://localhost:8081

## TODO

- [x] Submodule Chroma vector database
- [x] Build Flask interface for prototyping
- [x] Load Chroma with XOS Works embeddings
- [x] Get recommendations based on an ACMI collection Work
