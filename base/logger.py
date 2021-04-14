import logging
import logging.config

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            'format': '%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
        },
        'standard': {
            'format': '%(asctime)s [%(threadName)s:%(thread)d] [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "default": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": '/mnt/orangefs/Imagenet/tars_for_mul_proc/log.txt',
            'mode': 'w+',
            "maxBytes": 1024*1024*5,  # 5 MB
            "backupCount": 20,
            "encoding": "utf8"
        },
        "tars_to_h5s": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "",
            "filename": '/mnt/orangefs/Imagenet/tars_for_mul_proc/tars_to_h5s_{0}_used_time.txt',
            'mode': 'w+',
            "maxBytes": 1024*1024*10,  # 5 MB
            "backupCount": 20,
            "encoding": "utf8"
        },
    },

    "root": {
        'handlers': ['default'],
        'level': "INFO",
        'propagate': False
    }
}