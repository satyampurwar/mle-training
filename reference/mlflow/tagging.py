from datetime import datetime

from mlflow.tracking import MlflowClient

client = MlflowClient()
_run = client.get_run(run_id="993ac1452899493792a4440948c579c3")
dt = datetime.now().strftime("%d-%m-%Y (%H:%M:%S.%f)")
client.set_tag(_run.info.run_id, "deployed", dt)
