import uuid
from src.data_processor import DatasetBuilder
from src.model_manager import ModelLoader

class TrainingEngine:
    def __init__(self, config):
        self.config = config
        self.jobs = {}
        self.data_builder = DatasetBuilder(config)
        self.model_loader = ModelLoader(config)

    def create_job(self, job_config):
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": "queued", "config": job_config}
        return job_id

    def get_job_status(self, job_id):
        return self.jobs.get(job_id, {}).get("status", "not found")

    def run_job(self, job_id):
        job = self.jobs[job_id]
        job["status"] = "running"
        
        try:
            dataset = self.data_builder.load_data(job["config"]["dataset_path"])
            model = self.model_loader.load_base_model()
            model = self.model_loader.apply_lora(model)
            
            # Training logic here
            
            job["status"] = "completed"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
