import importlib

from .job_queue import JOB_QUEUE_NAME, _get_redis_connection


def main() -> None:
    rq_module = importlib.import_module("rq")
    worker = rq_module.Worker([JOB_QUEUE_NAME], connection=_get_redis_connection())
    worker.work()


if __name__ == "__main__":
    main()
