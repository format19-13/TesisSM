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

TRACK_TERMS = 'uruguay'

# Cuentas a escuchar: Lista de IDs de cuentas de Twitter en un string separado por comas
FOLLOWS_IDS = '2304818876, 74452681'

# Montevideo
BOUNDING_BOXES = [
                    -56.96411, -35.09295,
                    -53.53638, -33.68778
]

############################

PLACES_COUNTRY_IDS = ['UY']

# EXTRACTOR
CANT_WORKERS = 2


class EnumSource(Enum):
    GEOLOCATION = "geolocation"
    TRACKTERMS = "trackterms"
    FOLLOW = "follow"

SOURCE = [EnumSource.GEOLOCATION, EnumSource.TRACKTERMS, EnumSource.FOLLOW]

