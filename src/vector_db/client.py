from qdrant_client import QdrantClient

class VectorDBClient:
    def __init__(self, config):
        self.client = QdrantClient(url=config.vector_db_url)

    def create_collection(self, name, vector_size):
        self.client.create_collection(name, vector_size=vector_size)

    def insert_vectors(self, collection_name, vectors, payloads):
        self.client.upsert(collection_name, vectors=vectors, payloads=payloads)

    def search(self, collection_name, query_vector, limit=10):
        return self.client.search(collection_name, query_vector, limit=limit)
