from enum import StrEnum

class Colour(StrEnum):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def colour_this(cls, colour_this, *identifiers) -> str:
        return "".join(identifiers) + str(colour_this) + cls.ENDC.value

    @classmethod
    def bool_colour(cls, bool: bool, colour_this=None) -> str:
        if colour_this is None:
            colour_this = bool
        
        colour = cls.OKGREEN if bool else cls.FAIL
        return cls.BOLD + colour + str(colour_this) + cls.ENDC