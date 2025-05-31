from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    # api_key="<YOUR API KEY>" # optional to access your private data and models
)

result = client.run_workflow(
    workspace_name="roboflow-docs",
    workflow_id="model-comparison",
    images={
        "image": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
    },
    parameters={
        "model1": "yolov8n-640",
        "model2": "yolov11n-640"
    }
)

print(result)