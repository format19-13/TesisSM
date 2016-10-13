from enum import Enum

############################
# LOGGER
############################
LOGGING_BASE_PATH = '/var/log/twitter_streamer/'
LOGGING_ROOT_NAME = 'logfile'

############################

############################
# STREAMER
############################
TWITTER_ACCESS_KEYS = {
    "app_key": "HdmtYkvu7gqv64RAE6DTYhCYs",
    "app_secret": "F4NMWUsBgSnW4d4bzXsG2hOEFxBVXBhZHAim6bx8s7zvU5cjuV",
    "app_access_token": "780859625448939521-XZ0Pz1XH4fXBQuTZwYjoL49fTLMQSbK",
    "app_access_token_secret": "fiWErePI0a0KiKlwMvu9VRXXYN7ZdrAmFMiGhFouHsAWE",
    "description": "Twitter access key"
}

TRACK_TERMS = ''

# Cuentas a escuchar: Lista de IDs de cuentas de Twitter en un string separado por comas
USER_ID = '254142287'

# Montevideo
BOUNDING_BOXES = []

# Cuentas a escuchar: Lista de IDs de cuentas de Twitter en un string separado por comas
FOLLOWS_IDS = '2304818876, 74452681'

############################

PLACES_COUNTRY_IDS = []


class EnumSource(Enum):
    USER_ID = "user_id"

# EXTRACTOR
CANT_WORKERS = 2

SOURCE = [EnumSource.USER_ID]

