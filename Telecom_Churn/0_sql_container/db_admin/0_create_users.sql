-- create admin profile
CREATE ROLE admin WITH LOGIN PASSWORD 'admin';
GRANT ALL PRIVILEGES ON DATABASE churn_project TO admin;

-- create data scientist profile (view only)
CREATE ROLE ds_user WITH  LOGIN PASSWORD 'ds_user';
GRANT CONNECT ON DATABASE churn_project TO ds_user;
GRANT USAGE ON SCHEMA public TO ds_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ds_user;

-- add future table vew to ds_user, in case I add new tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ds_user;