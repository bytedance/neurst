#!/usr/bin/env bash
set -e

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"


if [[ -z ${NEURST_LIB} ]]
then
    NEURST_LIB=$THIS_DIR
    echo "using default --lib=${NEURST_LIB}" >&2
fi

pip3 install -e ${NEURST_LIB} --no-deps

if [[ $@ =~ "--enable_xla" ]]
then
    echo "enable XLA"
    export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
fi

python3 -m $@
