#!/bin/bash

# -------------- general approach start----------------

# 1. import this file: 
    # source path/change_color.sh
# 2. use COLOR/BG: 
    # $VARIABLE_NAME && out_put_content && $RESET
# 3. COLOR + BG:
    # $COLOR/BG_VARIABLE_NAME && $BG/COLOR_VARIABLE_NAME && out_put_content && $RESET
# 4. custom
    # abbreviation(change number)
        # txt number range (30, 37)
        # bg number range (40, 47)
        # special effects number range (1, 7)
        # echo -en \\E[number1 + ; + number2 + ; + number3 + m"
        # e.g - BG_GRAY+LIGHT_RED = "echo -en \\E[47;31m"

# -------------- general approach end----------------==


# general setting
# ------------- light_color start----------------
# black
LIGHT_BLACK="echo -en \\E[30m"
# red
LIGHT_RED="echo -en \\E[31m"
# green
LIGHT_GREEN="echo -en \\E[32m"
# yellow
LIGHT_YELLOW="echo -en \\E[33m"
# blue
LIGHT_BLUE="echo -en \\E[34m"
# purple
LIGHT_PURPLE="echo -en \\E[35m"
# cyan
LIGHT_CYAN="echo -en \\E[36m"
# gray
LIGHT_GRAY="echo -en \\E[37m"
# ------------- light_color end----------------

# ------------- bold_color start----------------
# black
BOLD_BLACK="echo -en \\E[1;30m"
# red
BOLD_RED="echo -en \\E[1;31m"
# green
BOLD_GREEN="echo -en \\E[1;32m"
# yellow
BOLD_YELLOW="echo -en \\E[1;33m"
# blue
BOLD_BLUE="echo -en \\E[1;34m"
# purple
BOLD_PURPLE="echo -en \\E[1;35m"
# cyan
BOLD_CYAN="echo -en \\E[1;36m"
# gray
BOLD_GRAY="echo -en \\E[1;37m"
# ------------- bold_color end----------------

# ------------- background_color start----------------
# black
BG_BLACK="echo -en \\E[40m"
# red
BG_RED="echo -en \\E[41m"
# green
BG_GREEN="echo -en \\E[42m"
# yellow
BG_YELLOW="echo -en \\E[43m"
# blue
BG_BLUE="echo -en \\E[44m"
# purple
BG_PURPLE="echo -en \\E[45m"
# cyan
BG_CYAN="echo -en \\E[46m"
# gray
BG_GRAY="echo -en \\E[47m"
# ------------- background_color end----------------

# close 
RESET="echo -en \\E[0m"
