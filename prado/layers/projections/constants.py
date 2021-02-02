import re

MEANINGLESS_PATTERN = r"^[^\w\d\!\$\%\&\?\.]+$"
MEANINGLESS_REGEX = re.compile(MEANINGLESS_PATTERN)
