from __future__ import annotations
import os
from pathlib import Path
from locust import HttpUser, between, task

class PetBreedInferenceUser(HttpUser):
    wait_time=between(0.1,0.5)
    payload_path=Path(os.getenv("PET_BREED_PAYLOAD","tests/fixtures/pet.jpg"))
    endpoint=os.getenv("PET_BREED_ENDPOINT","/predict")
    @task
    def predict(self):
        if not self.payload_path.exists(): raise RuntimeError(f"Payload image not found: {self.payload_path}")
        with self.payload_path.open("rb") as image:
            response=self.client.post(self.endpoint,files={"image":(self.payload_path.name,image,"image/jpeg")},name="POST /predict")
        if response.status_code>=400: response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
