import configparser
import time
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from categorize import categorize_question


config_path = Path(__file__).resolve().parent.parent / "config.ini"

if not config_path.exists():
    print("Please create config.ini file in the root directory")
    exit(1)

config = configparser.ConfigParser()
config.read(config_path)

app = FastAPI()


# get url as query parameter
@app.get("/categorize")
async def convert(filename: str, question_types: list(str)):  # response: int = 0
    start_time = time.time()
    result = categorize_question(filename, question_types)
    print(f"Time: {time.time() - start_time}")
    return result


if __name__ == "__main__":
    uvicorn.run(app, port=int(config["fastapi"]["port"]))
    # uvicorn.run(app, host=config["fastapi"]["host"], port=int(config["fastapi"]["port_categorize_question"])) # THIS PORT DOESN'T EXIST BUT IT IS JUST A PLACEHOLDER FOR NOW


