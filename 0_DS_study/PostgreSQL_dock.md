# PostgresSQL server using Docker
How to make a SQL container, using this [repository](https://github.com/docker/awesome-compose/tree/master/postgresql-pgadmin#add-postgres-database-to-pgadmin) as reference.

Official [PostgreSQL](https://hub.docker.com/_/postgres) guide.

## Summary
1. [Setting PostgreSQL](#seettingPost)
2. [PostgreSQL Image](#PostImage)

## Step-by-step

### Setting up PostgreSQL  <a name="seettingPost"></a>
Create a file called `.env` to set the basic features such as database name

```
POSTGRES_USER=yourUser
POSTGRES_PW=changeit
POSTGRES_DB=postgres
PGADMIN_MAIL=your@email.com
PGADMIN_PW=changeit
```

### PostgreSQL Image <a name="PostImage"></a>
In your VS Code, create a file with the following extension `.yml`. This is file tells docker the basic structure for your container, like a recipe.

1. create a file inside your project folder `docker-compose.yml`.

2. Fill `docker-compose.yml` with **PostgresSQL** 'recipe':
 
 ```
version: '3.8'

services:
  postgres:
    container_name: postgres
    image: postgres:latest  # build postgre image based on the latest version
    environment:
      POSTGRES_USER: {POSTGRES_USER}  # regular database username
      POSTGRES_PASSWORD: {POSTGRES_PW}    # regular database user password
    ports:
      - "5432:5432"     # postgre default port
    restart: always
    volumes:
      - postgres-data:/var/lib/postgresql/data     # save databases data evn when you turn the container off. 'postgres-data' is a hidden folder inside docker.

volumes:
    postgres-data:

  pgadmin:      # This creates a separate container for admins
    container_name: pgadmin
    image: dpage/pgadmin4:latest
    environment:
      - PGADMIN_DEFAULT_EMAIL=${PGADMIN_MAIL}
      - PGADMIN_DEFAULT_PASSWORD=${PGADMIN_PW}
    ports:
      - "5050:80"       # default admin port
    restart: always
 ```

3. Open your terminal and type: `docker-compose up -d`. 

Here you must use the name of `.yml` file, for example, if you named your yml file `composer.yml` then the command line would be `composer up -d`.

Here is important to note that docker needs to download the system image before it start the container. If this is the first time you are creating a containeer, it may take a while for docker to download it and push the server.