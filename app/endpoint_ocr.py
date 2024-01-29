import uvicorn
from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
import os
import datetime
from detectaicore import index_response, Job
import sys
import traceback
from dotenv import load_dotenv

try:
    from image.ocr.app.utils.ocr_c import (
        create_cffi,
        setup_path_library_names,
        initialize_tesseract_api,
    )
    from image.ocr.app.utils.utils import process_request
except:
    from utils.ocr_c import (
        create_cffi,
        setup_path_library_names,
        initialize_tesseract_api,
    )

    from utils.utils import process_request

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()


# load credentials
env_path = os.path.join("keys", ".env")
load_dotenv(env_path)
MODEL_PATH = os.getenv("MODEL_PATH")
LOCAL_ENV = os.getenv("LOCAL_ENV")

if MODEL_PATH == None and LOCAL_ENV == 0:
    MODEL_PATH = "/app/tessdata"
    app.tessdata = MODEL_PATH
    os.environ["TESSDATA_PREFIX"] = app.tessdata
else:
    # Use project tessdata
    app.tessdata = os.path.join(ROOT_DIR, "tessdata")
    os.environ["TESSDATA_PREFIX"] = app.tessdata

print(app.tessdata)
print(os.getenv("TESSDATA_PREFIX"))

# create cffi tesseract application
app.ffi = create_cffi()
app.tesseract, app.leptonica = setup_path_library_names(app.ffi)

# Create tesseract API
app.api = app.tesseract.TessBaseAPICreate()

# Initialize  Tesseract API
app.tesseract = initialize_tesseract_api(app.api, app.tesseract, app.tessdata)


# Create Jobs object
global jobs
jobs = {}


@app.get("/test")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"test endpoint is": "OK"})


@app.get("/health-check")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"message": "OK"})


@app.get("/work/status")
async def status_handler(request: Request):
    return jobs


@app.post("/process")
async def perform_ocr_c(request: Request, out: index_response):
    """
    Process Detect OCR extraction request
    """
    try:
        time1 = datetime.datetime.now()
        new_task = Job()
        # Capture Job and apply status
        jobs[new_task.uid] = new_task
        jobs[new_task.uid].status = "Job started"
        jobs[new_task.uid].type_job = "OCR Model Extraction"
        # get the base64 encoded string
        req = await request.json()

        print(os.getenv("TESSDATA_PREFIX"))

        if isinstance(req.get("documents"), list):
            list_docs = req.get("documents")
        else:
            raise AttributeError("Expected a list of Documents")
        # convert it into bytes
        cypher = req.get("cypher")

        # Extract response elements
        if isinstance(req.get("cypher"), int) or isinstance(req.get("cypher"), str):
            if isinstance(req.get("cypher"), str):
                cypher = int(req.get("cypher"))
            elif isinstance(req.get("cypher"), int):
                cypher = req.get("cypher")
            else:
                cypher = 0
        else:
            cypher = 0

        data, documents_non_teathred = await run_in_threadpool(
            process_request,
            list_docs=list_docs,
            app=app,
            jobs=jobs,
            new_task=new_task,
            cypher=cypher,
        )
        # Print whole recognized text

        time2 = datetime.datetime.now()
        t = time2 - time1
        tf = t.seconds * 1000000 + t.microseconds

        # create response
        out.status = {"code": 200, "message": "Success"}
        out.data = data
        out.number_documents_treated = len(data)
        out.number_documents_non_treated = len(documents_non_teathred)
        out.list_id_not_treated = documents_non_teathred

        json_compatible_item_data = jsonable_encoder(out)
        # Update jobs interface
        jobs[new_task.uid].status = f"Job {new_task.uid} Finished code {200}"
        return JSONResponse(content=json_compatible_item_data, status_code=200)

    except Exception as e:
        # cath exception with sys and return the error stack
        out.status = {"code": 500, "message": "Error"}
        ex_type, ex_value, ex_traceback = sys.exc_info()
        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # Format stacktrace
        stack_trace = list()

        for trace in trace_back:
            stack_trace.append(
                "File : %s , Line : %d, Func.Name : %s, Message : %s"
                % (trace[0], trace[1], trace[2], trace[3])
            )

        error = ex_type.__name__ + "\n" + str(ex_value) + "\n"
        for err in stack_trace:
            error = error + str(err) + "\n"
        out.error = error
        json_compatible_item_data = jsonable_encoder(out)
        return JSONResponse(content=json_compatible_item_data, status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        "endpoint_ocr:app",
        reload=True,
        host="0.0.0.0",
        port=5003,
        log_level="info",
    )
