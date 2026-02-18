import argparse
import asyncio
import logging

from indexer.pipeline import Pipeline, PipelineConfig


logger = logging.getLogger("indexer")
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

async def index() -> None:
    cfg = PipelineConfig()
    pipe = Pipeline(cfg)

    await pipe.initialize_db()
    stats = await pipe.run()
    logger.info("indexing finished: %s", stats)

    await pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="indexer")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run"])
    args = ap.parse_args()

    if args.cmd == "run":
        asyncio.run(index())


if __name__ == "__main__":
    main()
