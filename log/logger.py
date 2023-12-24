import logging


class Logger:
    def __init__(self, log_file='app.log'):
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)
        self.logger = logging.getLogger('MyLogger')

    def log_infor(self, message):
        self.logger.info(message)

    def log_debug(self, message):
        self.logger.debug(message)


