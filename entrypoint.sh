#!/bin/sh
# Start the facelive engine (also used by the Vault overlay, which runs it after the secrets are in place).
set -eu
exec python -m facelive
