import logging
import sys


class CustomLogger:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    def set_config(self, format, datefmt, handlers):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def set_training_args(self, training_args):
        # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
        if self.name == "__main__":
            self.logger.info("Training/evaluation parameters %s", training_args)
