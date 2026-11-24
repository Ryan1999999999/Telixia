#!/bin/bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v ro_pwd="$TELIXIA_RO_PASSWORD" <<-'EOSQL'
CREATE ROLE telixia_ro LOGIN PASSWORD :'ro_pwd';
SELECT format('GRANT CONNECT ON DATABASE %I TO telixia_ro', current_database()) \gexec
GRANT USAGE ON SCHEMA analytics, catalog, market TO telixia_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics, catalog, market TO telixia_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT ON TABLES TO telixia_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA catalog   GRANT SELECT ON TABLES TO telixia_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA market    GRANT SELECT ON TABLES TO telixia_ro;
ALTER ROLE telixia_ro SET default_transaction_read_only = on;
ALTER ROLE telixia_ro SET statement_timeout = '5s';
EOSQL

echo "telixia_ro read-only role created"