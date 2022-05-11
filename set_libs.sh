# Copyright 2022 Democritus University of Thrace
# Integrated Circuits Lab
# 
# Shell script to automatically download required libraries for Fast-Float4HLS examples

# Configure Fast-Float4HLS
if [ ! -d ./Fast-Float4HLS ]; then
  echo "Downloading Fast-Float4HLS..."
  git clone https://github.com/ic-lab-duth/Fast-Float4HLS.git
fi
FAST_FLOAT=`pwd`/Fast-Float4HLS
export FAST_FLOAT

# Configure AC Datatypes
if [ ! -d ./ac_types ]; then
  echo "Downloading AC_Types..."
  git clone http://github.com/hlslibs/ac_types.git
fi
AC_TYPES=`pwd`/ac_types
export AC_TYPES

# Configure AC Simutils
if [ ! -d ./ac_simutils ]; then
  echo "Downloading AC_Simutils..."
  git clone http://github.com/hlslibs/ac_simutils.git
fi
AC_SIMUTILS=`pwd`/ac_simutils
export AC_SIMUTILS