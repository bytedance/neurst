#!/usr/bin/env bash
set -e

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"


if [[ -z ${NEURST_LIB} ]]
then
    NEURST_LIB=$THIS_DIR
    echo "using default --lib=${NEURST_LIB}" >&2
fi

if [[ $@ =~ "--enable_xla" ]]
then
    export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
fi

neurst-run "$@"

