"""SDXL Turbo image generation server — drop-in replacement for Sana.

Uses SDXL Turbo (1-step) with TinyAutoencoder for maximum throughput.
Generates 512x512 images in ~50-200ms on consumer GPUs.
"""

import os, sys, io, base64, logging, torch, time, threading, warnings, aiohttp, asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
for noisy in ["httpx", "httpcore", "urllib3", "diffusers", "transformers", "huggingface_hub"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

MODEL_ID = "stabilityai/sdxl-turbo"
TAESD_ID = "madebyollin/taesdxl"
NUM_INFERENCE_STEPS = 2
MAX_DIM = 512

REGISTER_URL = os.environ.get("REGISTER_URL", "")
PUBLIC_IP = os.environ.get("PUBLIC_IP", "")
PUBLIC_PORT = os.environ.get("PUBLIC_PORT", "")
SERVICE_TYPE = os.environ.get("SERVICE_TYPE", "sana")
PORT = int(os.environ.get("PORT", "10003"))

generate_lock = threading.Lock()


class ImageRequest(BaseModel):
    prompts: list[str] = Field(default=["a cat"], min_length=1)
    width: int = Field(default=512)
    height: int = Field(default=512)
    seed: int | None = None
    steps: int | None = None
    safety_checker_adj: float | None = None


def clamp_dims(w, h):
    w, h = min(w, MAX_DIM), min(h, MAX_DIM)
    w = max(32, (w // 8) * 8)
    h = max(32, (h // 8) * 8)
    return w, h


pipe = None


async def send_heartbeat():
    """Register with the image service load balancer."""
    if not REGISTER_URL or not PUBLIC_IP:
        return
    url = f"http://{PUBLIC_IP}:{PUBLIC_PORT or PORT}"
    payload = {"url": url, "type": SERVICE_TYPE}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(REGISTER_URL, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    logger.debug("Heartbeat OK")
                else:
                    logger.warning("Heartbeat %d", resp.status)
    except Exception as e:
        logger.warning("Heartbeat failed: %s", e)


async def heartbeat_loop():
    while True:
        await send_heartbeat()
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    from diffusers import AutoPipelineForText2Image, AutoencoderTiny

    logger.info("Loading %s...", MODEL_ID)
    t0 = time.time()

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    # Use TinyAutoencoder for ~2x faster decoding
    pipe.vae = AutoencoderTiny.from_pretrained(
        TAESD_ID, torch_dtype=torch.float16
    ).to("cuda")

    # Warmup
    with torch.inference_mode():
        pipe("warmup", num_inference_steps=1, guidance_scale=0.0, width=512, height=512)

    logger.info("Model loaded + warmed up in %.1fs", time.time() - t0)

    # Start heartbeat
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    yield
    heartbeat_task.cancel()


app = FastAPI(title="SDXL-Turbo Legacy", lifespan=lifespan)


@app.post("/generate")
def generate(request: ImageRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(8), "big")
    generator = torch.Generator("cuda").manual_seed(seed)
    gen_w, gen_h = clamp_dims(request.width, request.height)
    steps = NUM_INFERENCE_STEPS  # SDXL Turbo is designed for 1-step; ignore client request

    try:
        t0 = time.time()
        with generate_lock:
            with torch.inference_mode():
                output = pipe(
                    prompt=request.prompts[0],
                    generator=generator,
                    width=gen_w,
                    height=gen_h,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                )
            image = output.images[0]

        elapsed = time.time() - t0
        logger.info("Generated %dx%d in %.3fs (steps=%d)", gen_w, gen_h, elapsed, steps)

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)

        return JSONResponse(content=[{
            "image": base64.b64encode(buf.getvalue()).decode(),
            "has_nsfw_concept": False,
            "concept": [],
            "width": image.width,
            "height": image.height,
            "seed": seed,
            "prompt": request.prompts[0],
        }])

    except torch.cuda.OutOfMemoryError as e:
        logger.error("OOM: %s", e)
        sys.exit(1)


@app.get("/health")
async def health():
    if pipe is None:
        raise HTTPException(status_code=503, detail="Not loaded")
    return {"status": "healthy", "model": MODEL_ID}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
